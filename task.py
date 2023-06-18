from celery import Celery

app = Celery('tasks', backend="mongodb+srv://AMalfez:AMalfez%402003@cluster0.h28yihv.mongodb.net/task",broker='tcp://localhost:5555')

@app.task
def add(x, y):
    return x + y