"""CLI entry point for cricket-analytics."""

import typer

app = typer.Typer(
    name="cricket-analytics",
    help="Predict cricket shot effectiveness from ball-by-ball data.",
)


@app.command()
def scrape(
    config: str = typer.Option("configs/scraper.yaml", help="Scraper config path"),
) -> None:
    """Scrape ball-by-ball data from ESPN Cricinfo."""
    typer.echo(f"Scraping with config: {config}")
    typer.echo("Scraper not yet implemented — see PR-2.")


@app.command()
def prepare(
    raw_dir: str = typer.Option("data/raw", help="Directory with raw scraped data"),
    output: str = typer.Option(
        "data/processed/features.parquet", help="Output feature file"
    ),
) -> None:
    """Run cleaning + feature engineering pipeline."""
    typer.echo(f"Preparing data: {raw_dir} -> {output}")
    typer.echo("Data pipeline not yet implemented — see PR-3.")


@app.command()
def train(
    features: str = typer.Option(
        "data/processed/features.parquet", help="Feature file path"
    ),
    config: str = typer.Option("configs/train.yaml", help="Training config path"),
    output_dir: str = typer.Option("models/", help="Directory to save trained models"),
) -> None:
    """Train all models and print comparison."""
    typer.echo(f"Training with features={features}, config={config}")
    typer.echo("Training pipeline not yet implemented — see PR-4/PR-5.")


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to saved model"),
    features: str = typer.Option(
        "data/processed/features.parquet", help="Feature file path"
    ),
) -> None:
    """Evaluate a saved model on the test set."""
    typer.echo(f"Evaluating model={model_path} on features={features}")
    typer.echo("Evaluation not yet implemented — see PR-4.")


if __name__ == "__main__":
    app()
