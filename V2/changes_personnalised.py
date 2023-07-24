   
class CacheResultsPersonnalised(typing.NamedTuple):
    """
    Model together with optional cached sales listing and user information.
    """

    model: sales_inference.types.Model

    recommendation: typing.Optional[sales_inference.types.SalesRecommendation]

    user_info: typing.Optional[sales_inference.types.UserRecord]


 # async def recommend(self,
    #                     ongoing: typing.List[str],
    #                     uid: str) -> sales_inference.types.SalesList:
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
        #     prior_recommendation = sales_inference.types.SalesRecommendation(
        #         reco=sales_inference.types.SalesList(top=[], bottom=[]),
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
        return sales_inference.types.UserRecord(category_1=0,
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
                type_=sales_inference.types.UserRecord,
                b=raw_user_info)
        else:
            user_info = self.get_cold_start_user()
        return CacheResultsPersonnalised(model, user_previous_reco, user_info)


    def create_scoring_input(self,
                             sale_info: typing.List[sales_inference.types.SaleRecord],
                             user_info: sales_inference.types.UserRecord,
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
                   sale_info: typing.List[sales_inference.types.SaleRecord],
                   user_info: sales_inference.types.UserRecord,
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
                           previous: sales_inference.types.SalesRecommendation,
                           model: sales_inference.types.Model,
                           child_specific: sales_inference.types.UserRecord
                           ) -> sales_inference.types.SalesRecommendation:
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
            return sales_inference.types.SalesRecommendation(
                reco=(sales_inference.types.SalesList(
                    top=[s for s in previous.reco.top if s in seen_top],
                    bottom=[s for s in previous.reco.bottom
                            if s in seen_bottom])),
                last_time=previous.last_time)

        # new user
        
        elif not seen:
            sorted_y = self.sort_sales(sale_info, child_specific, unseen)
            sorted_sales = [sublist[0] for sublist in sorted_y]
            sorted_sales.append('fcba9db5cae341cca6e6d3b7f')
            return sales_inference.types.SalesRecommendation(
                reco=(sales_inference.types.SalesList(top=sorted_sales[:2],
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
            
            return sales_inference.types.SalesRecommendation(
                reco=(sales_inference.types.SalesList(top=sorted_sales,
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



async def test(uid, ongoing):
    engine = Engine(redis_host='localhost',
                    redis_port=6379)
    # log.info('Started personalised sale feature flow')
    return await engine.recommend(uid=uid, ongoing=ongoing)


if __name__ == '__main__':
    ongoing = ['05ecc47f35924039b91d533e6', 'dbe64197c65f405894ecba9b1',
               'de30feee11504622b4ddd43e6', 'fe969c8841a24fe18ef8ac2f8',
               '4a116c9dcbc14d50883513d25', 'ec440dc35e54420fbd662e76b',
               '4774973de2654b2b9ed983a4c', 'ceefb43f79f346a6994aa06c4',
               '6d89a60b86144d9688dfb8993', '5744e96f79064b678d4adae43',
               'eae0f6408c5e4ce098eb18df6', '2dee57810ee44661a00a2123f',
               '83ecdda4e2fb42898ac039894', 'adcb0d2025ef4124a9cbd8ef0',
               'a9133e85859f40a2a1e174999', '10c7a7994e9644a6b73398043',
               'e0283ec59b7b4d7ead51ff640', '1b48587121134ac9adaf043c5',
               '3af469407f7e4033a94b9c16b', '7c0e1d547cd14f7b973d7ea71']
    asyncio.run(test(uid='TRibIhlHuGSAKTY4eul70SDmvNy2', ongoing=ongoing))