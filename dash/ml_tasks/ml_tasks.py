fromt .celery import celery

@app.task
def hello():
    print('Hello there')
    return 'Hello there'
