from celery import Celery

app = Celery('ml_tasks',
        broker = 'amqp://guest:guest@rabbit:5672/',
        backend = 'redis://red:6379/0',
        include=['ml_tasks.tasks'])

app.autodiscover_tasks()

app.conf.update(
        result_expires=3600,
        )


@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))

if __name__ == '__main__':
    app.start()
