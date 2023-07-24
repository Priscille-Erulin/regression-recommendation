import asyncio
import datetime
import operator
import typing
import tensorflow as tf

import pydantic
import pickle
import redis.asyncio

import type


EXPIRATION: typing.Final = datetime.timedelta(days=4)
regressor = pickle.load(open('MLPR_model.pkl', 'rb'))
# regressor = tf.keras.models.load_model('TF_model.h5')


class CacheResult(typing.NamedTuple):
    """
    Model together with an optional cached sales listing for user.
    """

    model: type.Model

    recommendation: typing.Optional[type.SalesRecommendation]

    child_specific: None


class CacheResultsWithAlerts(typing.NamedTuple):
    """
    Model together with optional cached sales listing and alerts for user.
    """
    model: type.Model

    recommendation: typing.Optional[type.SalesRecommendation]

    alerts: typing.Optional[type.Alerts]


class CacheResultsPersonnalised(typing.NamedTuple):
    """
    Model together with optional cached sales listing and user information.
    """

    model: type.Model

    recommendation: typing.Optional[type.SalesRecommendation]

    user_info: typing.Optional[type.UserRecord]


class Engine:
    """
    Sales recommendation engine.

    The model is simply persited to redis. That implies that the actual
    inference step just involves a simple lookup together with an update of
    user metadata to track the last visit of a user.
    """

    MODEL_KEY: typing.Final[str] = 'personnalised_sales_engine:sales'  # 'sales_engine:sales'

    USER_FMT: typing.Final[str] = 'reco:sales:{uid:s}'

    def __init__(self, redis_host: str, redis_port: int,
                 redis_pool_size: typing.Optional[int] = None):
        """
        Create a new engine instance for sales recommendation inference.

        :param redis_host: redis hostname or IP
        :param redis_port: redis TCP port
        :param redis_pool_size: maximum open connections to redis
        """
        self._redis = redis.asyncio.Redis(host=redis_host, port=redis_port,
                                          max_connections=redis_pool_size)

    async def _fetch_user_recommendation(self, uid: str) -> CacheResult:
        """
        Get a tuple of the model and previous listing of sales for a user.

        :param uid: user to get the listing for
        :return: tuple of model and previous recommendation for a user
        """
        pipeline = self._redis.pipeline()
        pipeline.get(self.MODEL_KEY)
        pipeline.get(self.USER_FMT.format(uid=uid))
        raw_model, raw_cached_recommendation = await pipeline.execute()
        model = pydantic.parse_raw_as(type.Model, raw_model)
        if raw_cached_recommendation:
            user_previous_recommendation = pydantic.parse_raw_as(
                type_=type.SalesRecommendation,
                b=raw_cached_recommendation,
            )
        else:
            user_previous_recommendation = None
        child_specific = None
        return CacheResult(model, user_previous_recommendation, child_specific)

    async def _put_user_recommendation(
            self,
            uid: str,
            recommendation: type.SalesRecommendation):
        """
        Put a recommendation result for a user to redis under an expirable key.

        :param uid: user id to associate recommendation with
        :param recommendation: ordered recommendation result
        """
        await self._redis.setex(
            name=self.USER_FMT.format(uid=uid),
            value=recommendation.json(),
            time=EXPIRATION,
        )

    def _n_to_top(self, top: typing.List[str], bottom: typing.List[str],
                  n: int) -> type.SalesList:
        """
        Create a top list with n elements, leaving the rest at the bottom.

        :param n: number of elements to bring to the top
        :param top: top list to take elements from
        :param bottom: bottom list to take elements from if top argument does
                       not hold enough elements
        :return: sales lists (top, bottom) where top holds max `n` elements
        """
        union = top + bottom
        return type.SalesList(
            top=union[:n],
            bottom=union[n:],
        )
    

    def _split_sales(self, ongoing: typing.List[str],
                     previous: type.SalesRecommendation):
        """Split ongoing sales into seen and unseen lists."""

        ongoing_ = set(ongoing)
        seen_top = set(previous.reco.top) & ongoing_
        seen_bottom = set(previous.reco.bottom) & ongoing_
        seen = seen_top | seen_bottom
        unseen = [s for s in ongoing if s not in seen]  # preserve order

        return seen_top, seen_bottom, seen, unseen


    def _end_with_choose(self, top: typing.List[str],
                         bottom: typing.List[str]
                         ) -> (typing.List[str], typing.List[str]):
        """
        Remove choose gift card and put it at the end of bottom.
        """
        return ([x for x in top if x != 'fcba9db5cae341cca6e6d3b7f'],
                ([x for x in bottom if x != 'fcba9db5cae341cca6e6d3b7f']
                 + ['fcba9db5cae341cca6e6d3b7f']))


    def _mk_recommendation(self, ongoing: typing.List[str],
                           previous: type.SalesRecommendation,
                           model: type.Model,
                           child_specific = None,
                           ) -> type.SalesRecommendation:
        """
        Make a sales recommendation list based on the current model.

        :param ongoing: list of on-going sales (sale ids)
        :param previous: previously given recommendation for a user
        :param model: recommendation as defined by the model
        :param child_specific: unused in the case because no user specific info
        is required
        :return: ordered sales recommendation
        """
        seen_top, seen_bottom, seen, unseen = self._split_sales(ongoing, previous)
        if not unseen:
            return type.SalesRecommendation(
                reco=self._n_to_top(
                    # Keep previous order of seen sales
                    top=[s for s in previous.reco.top if s in seen_top],
                    bottom=[s for s in previous.reco.bottom
                            if s in seen_bottom],
                    n=len(previous.reco.top),
                ),
                last_time=previous.last_time,
            )
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            new_visitor = not previous.reco.top and not previous.reco.bottom
            if new_visitor or now - previous.last_time > EXPIRATION:
                n = 2 + len([sale.sale_id for sale in model.sales
                             if sale.is_new])
                # top sales
                best_sales = sorted(model.sales,
                                    key=lambda x: x.score,
                                    reverse=True)[:2]
                print(best_sales)
                # new sales first
                other_sales = sorted(model.sales,
                                     key=lambda x: (x.is_new, x.score),
                                     reverse=True)
                model_listing = ([record.sale_id
                                  for record in best_sales] +
                                 [record.sale_id
                                  for record in other_sales
                                  if record not in best_sales]
                                 )
            elif len(unseen) == 1:
                n = len(seen_top) + 1
                model_listing = [record.sale_id for record in
                                 sorted(model.sales,
                                        key=operator.attrgetter('score'),
                                        reverse=True)]
            else:
                n = len(unseen)
                model_listing = [record.sale_id for record in
                                 sorted(model.sales,
                                        key=operator.attrgetter('score'),
                                        reverse=True)]
            top = ([s for s in unseen if s not in model_listing] +
                   [s for s in model_listing if s in unseen])
            # Seen sales at the bottom
            bottom = [
                s for s in (previous.reco.top + previous.reco.bottom)
                if s in seen
            ]
            top, bottom = self._end_with_choose(top, bottom)
            # Keep `n` sales in top section, rest on the bottom
            return type.SalesRecommendation(
                reco=self._n_to_top(top, bottom, n),
                last_time=now,
            )

    async def recommend(self, ongoing: typing.List[str],
                        uid: str) -> type.SalesList:
        """
        Make a listing of recommended sales for a given user.

        :param ongoing: list of on-going sales (sale ids)
        :param uid: user identifier
        :return: listing of sale ids, composed by top and bottom sub-lists
        """
        # In order to be a general function, it 
        cache_result = await self._fetch_user_recommendation(uid)
        model, prior_recommendation, child_specific = cache_result
        if not prior_recommendation:
            prior_recommendation = type.SalesRecommendation(
                reco=type.SalesList(top=[], bottom=[]),
                last_time=datetime.datetime.now(datetime.timezone.utc),
            )
        recommendation = self._mk_recommendation(
            ongoing=ongoing,
            previous=prior_recommendation,
            model=model,
            child_specific=child_specific
        )
        print(recommendation.reco)
        if recommendation.last_time != prior_recommendation.last_time:
            print("")
             # await self._put_user_recommendation(uid, recommendation)

        return recommendation.reco


class AlertsEngine(Engine):
    """
    Sales recommendation engine including alerts.

    The model inherits from the recommendation model above. On top of that,
    it extracts the alerts set by the userfrom redis and puts at the top of
    the recommendation, the unseen alerts.
    """

    ALERTS_FORMAT: typing.Final[str] = 'uwr:sales:{uid:s}'

    async def _fetch_user_recommendation(
            self,
            uid: str
            ) -> CacheResultsWithAlerts:
        """
        Get a tuple of the model, previous listing of sales and alerts
        set for a user.

        :param uid: user to get the listing and alerts for
        :return: tuple of model, previous recommendation and
        alerts for a user
        """
        pipeline = self._redis.pipeline()
        pipeline.get(self.MODEL_KEY)
        pipeline.get(self.USER_FMT.format(uid=uid))
        pipeline.get(self.ALERTS_FORMAT.format(uid=uid))
        (raw_model,
         raw_cached_recommendation,
         raw_cached_alerts
         ) = await pipeline.execute()
        model = pydantic.parse_raw_as(type.Model, raw_model)

        if raw_cached_recommendation:
            user_previous_recommendation = pydantic.parse_raw_as(
                type_=type.SalesRecommendation,
                b=raw_cached_recommendation,
            )
        else:
            user_previous_recommendation = None

        # condition for no alerts
        if raw_cached_alerts:
            user_alerts = pydantic.parse_raw_as(type.Alerts,
                                                raw_cached_alerts)
        else:
            user_alerts = None
        return CacheResultsWithAlerts(model,
                                      user_previous_recommendation,
                                      user_alerts)

    def _mk_recommendation(self, ongoing: typing.List[str],
                           previous: type.SalesRecommendation,
                           model: type.Model,
                           child_specific: type.Alerts
                           ) -> type.SalesRecommendation:
        """
        Make a sales recommendation by inheriting from
        the standard recommendation logic and
        adding the unseen alerts at the top.

        :param ongoing: list of on-going sales (sale ids)
        :param previous: previously given recommendation for a user
        :param model: recommendation as defined by the model
        :param child_specific: list of alerts set by the user
        :return: ordered sales recommendation
        """
        # inherits from the standard recommendation logic
        recommendation = super()._mk_recommendation(ongoing, previous, model)
        seen_sales = set(previous.reco.top) | set(previous.reco.bottom)
        seen_alerts = set(child_specific) & seen_sales
        unseen_alerts = [i for i in child_specific if i in ongoing if i not in
                         seen_alerts]

        # according to the product logic, we keep the same
        # number of sales in the top part
        n = len(recommendation.reco.top)
        now = datetime.datetime.now(datetime.timezone.utc)

        all_sales = recommendation.reco.top + recommendation.reco.bottom
        # ordered sales that are not unseen alerts
        not_alerts = [i for i in all_sales if i not in unseen_alerts]

        # places the unseen alerts at the beginning of the top list
        return type.SalesRecommendation(
            reco=self._n_to_top(unseen_alerts, not_alerts, n),
            last_time=now,
            )

    # async def recommend(self,
    #                     ongoing: typing.List[str],
    #                     uid: str) -> type.SalesList:
        """
        Make a listing of recommended sales for a given user with
        the alerts logic in place.

        :param ongoing: list of on-going sales (sale ids)
        :param uid: user identifier
        :return: listing of sale ids, composed by top and bottom sub-lists
        with any unseen alerts at the beginning of top
        """
        # cache_result = await self._fetch_user_recommendation(uid)
        # model, prior_recommendation, set_alerts = cache_result

        # # changes the prior recommendations
        # # in an empty SalesRecommendation instead
        # # of None if there are no prior recommendations
        # if not prior_recommendation:
        #     prior_recommendation = type.SalesRecommendation(
        #         reco=type.SalesList(top=[], bottom=[]),
        #         last_time=datetime.datetime.now(datetime.timezone.utc),
        #     )
        # # changes the alerts in an empty
        # # list instead of None if there are no alerts
        # if not set_alerts:
        #     set_alerts = []

        # # sorted by putting unseen alerts at the beginning of the top sub-list
        # sorted_recommendation = self._mk_recommendation(
        #     ongoing=ongoing,
        #     previous=prior_recommendation,
        #     model=model,
        #     alerts=set_alerts,
        # )
        # # puts the new recommendation in the cache
        # if sorted_recommendation.last_time != prior_recommendation.last_time:
        #     await self._put_user_recommendation(uid, sorted_recommendation)

        # return sorted_recommendation.reco


class PersonalisedEngine(Engine):

    USER_INFO: typing.Final[str] = 'user:info:{uid:s}'

    PRED_FEATURES= ['log_delta','log_followers','conversion',
                     'log_revenue','log_brand_appearance','log_avg_price',
                     'artisanal', 'b_corporation', 'bio', 'biodegradable',
                     'cadeau_ideal', 'concept_original', 'durable',
                     'eco_friendly','excellent_sur_yuka', 'exclusivite_choose',
                     'fabrication_a_la_demande', 'fait_main', 'gluten_free',
                     'iconique', 'inclusive', 'innovation', 'made_in_europe',
                     'made_in_france', 'madeinjapan', 'naturel', 'oeko_tex',
                     'premium', 'recyclable', 'saint_valentin', 'savoir_faire',
                     'seconde_main', 'socialement_engagee', 'serie_limitee',
                     'tendance', 'upcycling', 'vegan', 'vintage', 'zerodechet',
                     'category_sale','log_monetary', 'log_frequency','log_recency',
                     'category_1','category_2', 'category_3'
                     ]


    def get_cold_start_user(self):
        """Return user features for new user."""
        updated_at = (datetime.datetime.utcnow()
                      .isoformat(sep="T", timespec="milliseconds")
                      + "Z")
        return type.UserRecord(category_1=0,
                                                category_2=0,
                                                category_3=0,
                                                log_monetary=0.0,
                                                log_frequency=0.0,
                                                log_recency=0.0,
                                                updated_at=updated_at)


    async def _fetch_user_recommendation(self,uid: str
                                         ) -> CacheResultsPersonnalised:
        """ Get sale, user and previous recommendation from Redis"""
        cache_result = await super()._fetch_user_recommendation(uid)
        model, user_previous_reco = cache_result.model, cache_result.recommendation
        raw_user_info = await self._redis.get(self.USER_INFO.format(uid=uid))

        if raw_user_info:
            user_info = pydantic.parse_raw_as(
                type_=type.UserRecord,
                b=raw_user_info)
        else:
            user_info = self.get_cold_start_user()
        return CacheResultsPersonnalised(model, user_previous_reco, user_info)


    def create_scoring_input(self,
                             sale_info: typing.List[type.SaleRecord],
                             user_info: type.UserRecord,
                             unseen: typing.List[str],
                             pred_features = PRED_FEATURES):
        """
        Create array for scoring and a list of sale_ids.

        :param sale_info: sales and sale features
        :param user_info: user features
        :param unseen: list of sale ids of sales unseen by user
        :param pred_features: list of features used by ML model
        """
        # the last value of of user_info is the time of update
        X_user = [v for k,v in user_info.dict().items() if k != 'updated_at']
        filtered_ongoing = [d for d in sale_info if d.sale_id in unseen]
        X = [[v for k,v in f.dict().items() if k in pred_features]
             + X_user for f in filtered_ongoing]
        # X = [list(d.__dict__.values())[1:] + X_user for d in filtered_ongoing]
        sale_ids = [f.sale_id for f in filtered_ongoing]
        print('hello')
        print([f.category_sale for f in filtered_ongoing])
        return X, sale_ids

    def sort_sales(self,
                   sale_info: typing.List[type.SaleRecord],
                   user_info: type.UserRecord,
                   unseen: typing.List[str]) -> typing.List[typing.List[str]]:
        """
        Make predictions and sort sale ids according to score.

        :param ongoing: on-going sale ids and features
        :param user_info: user features
        :param unseen: list of sale ids of sales unseen by user
        """
        X, sale_ids = self.create_scoring_input(sale_info, user_info, unseen)
        if X == []:
            return []
        else:
            y = regressor.predict(X)
            scored_sales = [[sale_ids[i], y[i]] for i in range(len(y))]
            return sorted(scored_sales, key=lambda x: x[1], reverse=True)

    def _mk_recommendation(self, ongoing: typing.List[str],
                           previous: type.SalesRecommendation,
                           model: type.Model,
                           child_specific: type.UserRecord
                           ) -> type.SalesRecommendation:
        """
        Make a sales recommendation list based on the current model.

        :param ongoing: on-going sale ids
        :param previous: previously given recommendation for a user
        :param model: sale features and update date
        :param child_specific: user features used for prediction
        :return: ordered sales recommendation
        """
        seen_top, seen_bottom, seen, unseen = super()._split_sales(ongoing, previous)
        sale_info = model.sales
        now = datetime.datetime.now(datetime.timezone.utc)
        # all sales have been seen
        print(child_specific)
        print(unseen)
        if not unseen:
            return type.SalesRecommendation(
                reco=(type.SalesList(
                    top=[s for s in previous.reco.top if s in seen_top],
                    bottom=[s for s in previous.reco.bottom
                            if s in seen_bottom])),
                last_time=previous.last_time)

        # new user
        
        elif not seen:
            sorted_y = self.sort_sales(sale_info, child_specific, unseen)
            sorted_sales = [sublist[0] for sublist in sorted_y]
            sorted_sales.append('fcba9db5cae341cca6e6d3b7f')
            return type.SalesRecommendation(
                reco=(type.SalesList(top=sorted_sales[:2],
                                                bottom=sorted_sales[2:])),
                last_time=now)

        else:
            sorted_y = self.sort_sales(sale_info, child_specific, unseen)
            sorted_sales = [sublist[0] for sublist in sorted_y]
            # make sure that choose is in last position
            bottom = [
                s for s in (previous.reco.top + previous.reco.bottom)
                if s in seen]
            if 'fcba9db5cae341cca6e6d3b7f' in bottom:
                bottom.remove('fcba9db5cae341cca6e6d3b7f')
            bottom.append('fcba9db5cae341cca6e6d3b7f')
            
            return type.SalesRecommendation(
                reco=(type.SalesList(top=sorted_sales,
                                                bottom=bottom)),
                last_time=now)

async def test(uid, ongoing):
    engine = PersonalisedEngine(
    redis_host='localhost',
    redis_port=6379)
    # log.info('Started personalised sale feature flow')
    return await engine.recommend(uid=uid, ongoing=ongoing)


if __name__ == '__main__':
    ongoing = ['05ecc47f35924039b91d533e6','dbe64197c65f405894ecba9b1', 'de30feee11504622b4ddd43e6', 'fe969c8841a24fe18ef8ac2f8', '4a116c9dcbc14d50883513d25',
               'ec440dc35e54420fbd662e76b', '4774973de2654b2b9ed983a4c', 'ceefb43f79f346a6994aa06c4', '6d89a60b86144d9688dfb8993', '5744e96f79064b678d4adae43',
               'eae0f6408c5e4ce098eb18df6', '2dee57810ee44661a00a2123f', '83ecdda4e2fb42898ac039894', 'adcb0d2025ef4124a9cbd8ef0', 'a9133e85859f40a2a1e174999',
               '10c7a7994e9644a6b73398043', 'e0283ec59b7b4d7ead51ff640', '1b48587121134ac9adaf043c5', '3af469407f7e4033a94b9c16b', '7c0e1d547cd14f7b973d7ea71']
    asyncio.run(test(uid='TRibIhlHuGSAKTY4eul70SDmvNy2', ongoing=ongoing))