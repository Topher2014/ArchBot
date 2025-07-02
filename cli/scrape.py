"""
Scraping commands for RDB CLI.
"""

import click
from datetime import datetime

from rdb.scraper.wiki_scraper import WikiScraper
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer


@click.command()
@click.option('--output', '-o', help='Output directory for scraped data')
@click.option('--delay-min', type=float, help='Minimum delay between requests (seconds)')
@click.option('--delay-max', type=float, help='Maximum delay between requests (seconds)')
@click.option('--max-retries', type=int, help='Maximum retries for failed requests')
@click.option('--resume', is_flag=True, help='Resume from existing scrape (skip existing files)')
@click.option('--force', is_flag=True, help='Force re-scrape all pages')
@click.option('--history', is_flag=True, help='Show scraping history')
@click.option('--limit', '-l', type=int, default=10, help='Number of recent sessions to show (with --history)')
@click.pass_context
def scrape_cmd(ctx, output, delay_min, delay_max, max_retries, resume, force, history, limit):
    """Scrape Arch Wiki documentation."""
    config = ctx.obj['config']
    
    # Handle history command
    if history:
        db_manager = DatabaseManager(config)
        stats = db_manager.get_scraping_stats()
        
        click.echo("Scraping Statistics:")
        click.echo(f"  Total sessions: {stats['total_sessions']}")
        click.echo(f"  Total pages scraped: {stats['total_pages_scraped']}")
        click.echo(f"  Total errors: {stats['total_errors']}")
        click.echo(f"  Last scrape: {stats['last_scrape']}")
        return
    
    # Override config with command line options
    if delay_min is not None:
        config.scrape_delay_min = delay_min
    if delay_max is not None:
        config.scrape_delay_max = delay_max
    if max_retries is not None:
        config.scrape_max_retries = max_retries
    
    # Initialize scraper
    scraper = WikiScraper(config)
    db_manager = DatabaseManager(config)
    
    click.echo("Starting Arch Wiki scraping...")
    click.echo(f"Output directory: {output or config.raw_data_dir}")
    click.echo(f"Delay range: {config.scrape_delay_min}-{config.scrape_delay_max}s")
    
    if not resume and not force:
        click.echo("\nThis will scrape the entire Arch Wiki.")
        click.echo("This may take several hours and will make many HTTP requests.")
        if not click.confirm("Do you want to continue?"):
            click.echo("Scraping cancelled.")
            return
    
    # Clear existing data if force is specified
    if force and output is None:
        import shutil
        if config.raw_data_dir.exists():
            click.echo("Removing existing scraped data...")
            shutil.rmtree(config.raw_data_dir)
            config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Start scraping
    session_data = {
        'started_at': datetime.now(),
        'config': {
            'delay_min': config.scrape_delay_min,
            'delay_max': config.scrape_delay_max,
            'max_retries': config.scrape_max_retries,
            'output_dir': str(output or config.raw_data_dir)
        }
    }
    
    try:
        with Timer("Scraping") as timer:
            success_count = scraper.scrape_all(output)
        
        session_data.update({
            'completed_at': datetime.now(),
            'total_pages': success_count,
            'success_count': success_count,
            'error_count': 0,
            'skip_count': 0,
            'status': 'completed'
        })
        
        # Log session to database
        session_id = db_manager.log_scraping_session(session_data)
        
        click.echo(f"\nScraping completed successfully!")
        click.echo(f"Pages scraped: {success_count}")
        click.echo(f"Time taken: {timer}")
        click.echo(f"Session ID: {session_id}")
        
    except KeyboardInterrupt:
        session_data.update({
            'completed_at': datetime.now(),
            'status': 'interrupted'
        })
        db_manager.log_scraping_session(session_data)
        
        click.echo("\nScraping interrupted by user.")
        
    except Exception as e:
        session_data.update({
            'completed_at': datetime.now(),
            'status': 'failed',
            'error': str(e)
        })
        db_manager.log_scraping_session(session_data)
        
        click.echo(f"\nScraping failed: {e}")
        raise click.ClickException(str(e))
