"""
Main CLI entry point for RDB.
"""

import click
import sys
from pathlib import Path

# Add the parent directory to the path so we can import rdb
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdb.config.settings import Config
from rdb.utils.logging import setup_logging
from .scrape import scrape_cmd
from .build import build_cmd
from .search import search_cmd


@click.group()
@click.option('--data-dir', '-d', help='Data directory path')
@click.option('--log-level', '-l', default='INFO', help='Logging level')
@click.option('--log-file', help='Log file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, data_dir, log_level, log_file, verbose):
   """RDB - Retrieval Database for Arch Wiki Documentation"""
   
   # Setup logging
   if verbose:
       log_level = 'DEBUG'
   
   setup_logging(log_level=log_level, log_file=log_file)
   
   # Initialize config
   config = Config(data_dir=data_dir)
   
   # Store config in context for subcommands
   ctx.ensure_object(dict)
   ctx.obj['config'] = config


@cli.command()
@click.pass_context
def version(ctx):
   """Show version information."""
   from rdb import __version__, __author__
   
   click.echo(f"RDB version {__version__}")
   click.echo(f"Author: {__author__}")


@cli.command()
@click.pass_context
def status(ctx):
   """Show RDB status and configuration."""
   config = ctx.obj['config']
   
   click.echo("RDB Status:")
   click.echo(f"  Data directory: {config.data_dir}")
   click.echo(f"  Raw data: {config.raw_data_dir}")
   click.echo(f"  Chunks: {config.chunks_dir}")
   click.echo(f"  Index: {config.index_dir}")
   click.echo(f"  Cache: {config.cache_dir}")
   click.echo()
   
   # Check if files exist
   click.echo("Data Status:")
   
   if config.raw_data_dir.exists():
       json_files = list(config.raw_data_dir.glob("*.json"))
       click.echo(f"  Raw data files: {len(json_files)}")
   else:
       click.echo("  Raw data files: Not found")
   
   if config.chunks_file.exists():
       click.echo(f"  Chunks file: ✓ {config.chunks_file}")
   else:
       click.echo("  Chunks file: ✗ Not found")
   
   if config.index_file.exists():
       click.echo(f"  Index file: ✓ {config.index_file}")
   else:
       click.echo("  Index file: ✗ Not found")
   
   if config.metadata_file.exists():
       click.echo(f"  Metadata file: ✓ {config.metadata_file}")
   else:
       click.echo("  Metadata file: ✗ Not found")


# Add subcommands
cli.add_command(scrape_cmd, name='scrape')
cli.add_command(build_cmd, name='build')
cli.add_command(search_cmd, name='search')


def main():
   """Main entry point for the CLI."""
   cli()


if __name__ == '__main__':
   main()
