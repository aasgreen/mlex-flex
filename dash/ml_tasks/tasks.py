from .celery import app as celery_app

@celery_app.task
def hello():
    print('Hello there')
    return 'Hello there'
