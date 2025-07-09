"""
Search commands for RDB CLI.
"""

import click
import time
from datetime import datetime

from rdb.retrieval.retriever import DocumentRetriever
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer


@click.command()
@click.argument('query', required=False)
@click.option('--top-k', '-k', type=int, help='Number of results to return')
@click.option('--refine', '-r', is_flag=True, help='Enable query refinement')
@click.option('--no-refine', is_flag=True, help='Disable query refinement')
@click.option('--refiner-model', '-m', help='Specify refiner model (path or HuggingFace ID)')  # <-- ADD THIS
@click.option('--interactive', '-i', is_flag=True, help='Start interactive search mode')
@click.option('--max-content', type=int, default=300, help='Maximum content length to display')
@click.option('--show-refinement', is_flag=True, help='Show query refinement process')
@click.option('--history', is_flag=True, help='Show search history')
@click.option('--limit', '-l', type=int, default=20, help='Number of recent searches to show (with --history)')
@click.pass_context
def search_cmd(ctx, query, top_k, refine, no_refine, refiner_model, interactive, max_content, show_refinement, history, limit):
    """Search the RDB index."""
    config = ctx.obj['config']
    
    # Handle history command
    if history:
        db_manager = DatabaseManager(config)
        
        # Get search statistics
        stats = db_manager.get_search_stats()
        
        click.echo("Search Statistics:")
        click.echo(f"  Total searches: {stats['total_searches']}")
        click.echo(f"  Average search time: {stats['avg_search_time_ms']:.1f}ms")
        click.echo(f"  Searches with refinement: {stats['refined_searches']}")
        click.echo(f"  Last search: {stats['last_search']}")
        
        # Get recent searches
        recent_searches = db_manager.get_recent_searches(limit)
        
        if recent_searches:
            click.echo(f"\nRecent Searches (last {len(recent_searches)}):")
            for search in recent_searches:
                timestamp = search['timestamp']
                original = search['original_query']
                refined = search['refined_query']
                results = search['results_count']
                time_ms = search['search_time_ms']
                
                click.echo(f"  {timestamp}")
                click.echo(f"    Query: {original}")
                if refined and refined != original:
                    click.echo(f"    Refined: {refined}")
                click.echo(f"    Results: {results}, Time: {time_ms}ms")
                click.echo()
        return
    
    # Override config with command line options
    if top_k:
        config.default_top_k = top_k

    # Override refiner model if specified
    if refiner_model:
        config.refiner_model = refiner_model
        # Force enable refinement if model is specified
        if not no_refine:
            refine = True
    
    # Determine refinement setting
    use_refinement = config.enable_query_refinement
    if refine or refiner_model:
        use_refinement = True
    elif no_refine:
        use_refinement = False
    
    try:
        # Temporarily override config based on CLI flag
        original_refinement_setting = config.enable_query_refinement
        config.enable_query_refinement = use_refinement
        
        retriever = DocumentRetriever(config)
        if not retriever.load_index():
            raise click.ClickException("Could not load search index. Run 'rdb build' first.")
        
        # Restore original config setting
        config.enable_query_refinement = original_refinement_setting
        
    except Exception as e:
        raise click.ClickException(f"Error initializing retriever: {e}")
    
    if interactive or not query:
        # Interactive mode
        retriever.search_interactive(top_k=config.default_top_k, show_refinement=show_refinement)
    else:
        # Single query mode
        db_manager = DatabaseManager(config)
        
        click.echo(f"Searching for: '{query}'")
        
        search_data = {
            'original_query': query,
            'top_k': config.default_top_k,
            'query_refinement_enabled': use_refinement
        }
        
        try:
            with Timer() as timer:
                results = retriever.search(
                    query, 
                    top_k=config.default_top_k, 
                    refine_query=use_refinement,
                    show_refinement=show_refinement
                )
            
            search_data.update({
                'refined_query': results[0]['final_query'] if results else query,
                'results_count': len(results),
                'search_time_ms': int(timer.elapsed * 1000)
            })
            
            # Log search to database
            db_manager.log_search(search_data)
            
            if not results:
                click.echo("No results found.")
            else:
                _print_results(results, max_content_length=max_content, show_queries=show_refinement)
                
                click.echo(f"\nSearch completed in {timer}")
                
        except Exception as e:
            search_data.update({
                'results_count': 0,
                'search_time_ms': 0
            })
            db_manager.log_search(search_data)
            
            raise click.ClickException(f"Search failed: {e}")


def _print_results(results, max_content_length=300, show_queries=False):
    """Print search results in a formatted way."""
    if show_queries and results:
        click.echo(f"\nOriginal query: {results[0]['original_query']}")
        if results[0]['original_query'] != results[0]['final_query']:
            click.echo(f"Refined query:  {results[0]['final_query']}")
    
    for result in results:
        click.echo(f"\n{'='*60}")
        click.echo(f"Rank {result['rank']} | Score: {result['score']:.4f}")
        click.echo(f"Page: {result['page_title']}")
        click.echo(f"Section: {result['section_path']}")
        click.echo(f"Type: {result['chunk_type']}")
        click.echo(f"URL: {result['url']}")
        click.echo("-" * 60)
        
        # Truncate content for display
        content = result['content']
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        click.echo(content)
