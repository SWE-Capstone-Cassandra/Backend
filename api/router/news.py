from fastapi import APIRouter


from service.news_service import NewsService

news_router = APIRouter()


@news_router.get("/{news_id}")
def get_news_by_news_id(news_id: int):
    return NewsService().get_news_data(news_id=news_id)


@news_router.get("/list/{item_name}/{page}")
def get_news_list_by_name(item_name: str, page: int):
    return NewsService().get_news_list(item_name=item_name, page=page)
