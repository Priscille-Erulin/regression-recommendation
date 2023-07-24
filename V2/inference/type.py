
import datetime
import typing

import pydantic


class SalesList(pydantic.BaseModel):
    """
    Ordered list of sale ids that are split into a top view, and bottom view.
    """
    top: typing.List[str]
    bottom: typing.List[str]


class SalesRecommendation(pydantic.BaseModel):
    """
    Sales recommendation for a user.
    """
    last_time: datetime.datetime
    reco: SalesList


class SaleRecord(pydantic.BaseModel):
    sale_id: str
    brand: str
    score: float
    is_new: bool
    log_delta: float
    log_followers: float
    conversion: float
    log_revenue: float
    log_brand_appearance: float
    log_avg_price: float
    artisanal: int
    b_corporation: int
    bio: int
    biodegradable: int
    cadeau_ideal: int
    concept_original: int
    durable: int
    eco_friendly: int
    excellent_sur_yuka: int
    exclusivite_choose: int
    fabrication_a_la_demande: int
    fait_main: int
    gluten_free: int
    iconique: int
    inclusive: int
    innovation: int
    made_in_europe: int
    made_in_france: int
    madeinjapan: int
    naturel: int
    oeko_tex: int
    premium: int
    recyclable: int
    saint_valentin: int
    savoir_faire: int
    seconde_main: int
    socialement_engagee: int
    serie_limitee: int
    tendance: int
    upcycling: int
    vegan: int
    vintage: int
    zerodechet: int
    category_sale: int


class Model(pydantic.BaseModel):
    sales: typing.List[SaleRecord]
    updated_at: datetime.datetime


Alerts = typing.List[str]

class UserRecord(pydantic.BaseModel):
    """Base class for sale records."""
    log_monetary: float
    log_frequency: float
    log_recency: float
    category_1: int
    category_2: int
    category_3: int
    updated_at: str
