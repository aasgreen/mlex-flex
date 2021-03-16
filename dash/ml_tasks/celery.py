from celery import Celery

celery = Celery('ml_tasks',
        broker = 'amqp://'
        backend = 'redis://'
        include=['ml_tasks.tasks'])

celer.autodiscover_tasks()

celery.conf.update(
        result_expires=3600,
        )


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))

if __name__ == '__main__':
    celery.start()
