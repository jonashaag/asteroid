import typer

app = typer.Typer()

from asteroid.cli.train import app as train_app

app.add_typer(train_app)
