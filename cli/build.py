"""
Index building commands for RDB CLI.
"""

import click
from datetime import datetime

from rdb.chunking.chunker import DocumentChunker
from rdb.embedding.embedder import DocumentEmbedder
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer


@click.command()
@click.option('--input', '-i', help='Input directory with scraped JSON files')
@click.option('--output', '-o', help='Output directory for index files')
@click.option('--embedding-model', '-m', help='Embedding model to use')
@click.option('--batch-size', '-b', type=int, help='Batch size for embedding creation')
@click.option('--force', is_flag=True, help='Force rebuild even if index exists')
@click.option('--stats', is_flag=True, help='Show index statistics')
@click.pass_context
def build_cmd(ctx, input, output, embedding_model, batch_size, force, stats):
    """Build search index from scraped data."""
    config = ctx.obj['config']
    
    # Handle stats command
    if stats:
        click.echo("Index Status:")
        
        if config.index_file.exists():
            try:
                import faiss
                index = faiss.read_index(str(config.index_file))
                click.echo(f"  Index file: ✓ {config.index_file}")
                click.echo(f"  Total vectors: {index.ntotal}")
                click.echo(f"  Vector dimension: {index.d}")
                click.echo(f"  Index type: {type(index).__name__}")
            except Exception as e:
                click.echo(f"  Index file: ✗ Error loading: {e}")
        else:
            click.echo(f"  Index file: ✗ Not found")
        
        if config.metadata_file.exists():
            try:
                import pickle
                with open(config.metadata_file, 'rb') as f:
                    chunks = pickle.load(f)
                click.echo(f"  Metadata file: ✓ {config.metadata_file}")
                click.echo(f"  Total chunks: {len(chunks)}")
                
                # Chunk type distribution
                chunk_types = {}
                for chunk in chunks:
                    chunk_type = chunk.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                click.echo("  Chunk distribution:")
                for chunk_type, count in chunk_types.items():
                    click.echo(f"    {chunk_type}: {count}")
                    
            except Exception as e:
                click.echo(f"  Metadata file: ✗ Error loading: {e}")
        else:
            click.echo(f"  Metadata file: ✗ Not found")
        return
    
    # Override config with command line options
    if embedding_model:
        config.embedding_model = embedding_model
    if batch_size:
        config.embedding_batch_size = batch_size
    
    input_dir = input or config.raw_data_dir
    output_dir = output or config.index_dir
    
    click.echo("Building search index...")
    click.echo(f"Input directory: {input_dir}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Embedding model: {config.embedding_model}")
    click.echo(f"Batch size: {config.embedding_batch_size}")
    
    # Check if index already exists
    if not force and config.index_file.exists():
        if not click.confirm(f"Index already exists at {config.index_file}. Rebuild?"):
            click.echo("Build cancelled.")
            return
    
    # Initialize components
    chunker = DocumentChunker(config)
    embedder = DocumentEmbedder(config)
    db_manager = DatabaseManager(config)
    
    session_data = {
        'started_at': datetime.now(),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'embedding_model': config.embedding_model,
        'config': {
            'embedding_model': config.embedding_model,
            'batch_size': config.embedding_batch_size,
            'chunk_size_small': config.chunk_size_small,
            'chunk_size_medium': config.chunk_size_medium,
            'chunk_size_large': config.chunk_size_large
        }
    }
    
    try:
        with Timer("Index building") as timer:
            # Step 1: Process documents into chunks
            click.echo("\nStep 1: Processing documents into chunks...")
            with Timer("Chunking") as chunk_timer:
                chunks = chunker.process_directory(input_dir)
                chunker.save_chunks()
            
            click.echo(f"Created {len(chunks)} chunks in {chunk_timer}")
            chunker.print_stats()
            
            # Step 2: Create embeddings
            click.echo("\nStep 2: Creating embeddings...")
            with Timer("Embedding creation") as embed_timer:
                embeddings = embedder.create_embeddings(chunks)
            
            click.echo(f"Created embeddings in {embed_timer}")
            
            # Step 3: Build index
            click.echo("\nStep 3: Building FAISS index...")
            with Timer("Index building") as index_timer:
                embedder.build_index(embeddings)
                index_file, metadata_file = embedder.save_index(output_dir)
            
            click.echo(f"Built index in {index_timer}")
        
        session_data.update({
            'completed_at': datetime.now(),
            'total_chunks': len(chunks),
            'status': 'completed'
        })
        
        # Log session to database
        session_id = db_manager.log_indexing_session(session_data)
        
        click.echo(f"\nIndex building completed successfully!")
        click.echo(f"Total chunks: {len(chunks)}")
        click.echo(f"Index file: {index_file}")
        click.echo(f"Metadata file: {metadata_file}")
        click.echo(f"Time taken: {timer}")
        click.echo(f"Session ID: {session_id}")
        
    except KeyboardInterrupt:
        session_data.update({
            'completed_at': datetime.now(),
            'status': 'interrupted'
        })
        db_manager.log_indexing_session(session_data)
        
        click.echo("\nIndex building interrupted by user.")
        
    except Exception as e:
        session_data.update({
            'completed_at': datetime.now(),
            'status': 'failed',
            'error': str(e)
        })
        db_manager.log_indexing_session(session_data)
        
        click.echo(f"\nIndex building failed: {e}")
        raise click.ClickException(str(e))
