"""
Search API routes.
"""

import json
import logging
import io
import sys
import time
from flask import Blueprint, request, jsonify, current_app, Response
from rdb.retrieval.retriever import DocumentRetriever
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer

search_bp = Blueprint('search', __name__)
logger = logging.getLogger(__name__)


class LogCapture:
    """Capture logs and yield them for SSE streaming."""
    
    def __init__(self):
        self.logs = []
        self.handler = None
    
    def start_capture(self):
        """Start capturing logs."""
        self.handler = logging.StreamHandler(io.StringIO())
        self.handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        
        # Add handler to root logger to capture all logs
        logging.getLogger().addHandler(self.handler)
    
    def stop_capture(self):
        """Stop capturing logs."""
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
    
    def get_new_logs(self):
        """Get new log entries since last call."""
        if not self.handler:
            return []
        
        # Get the captured content
        content = self.handler.stream.getvalue()
        
        # Split into lines and filter out empty ones
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Reset the stream
        self.handler.stream.seek(0)
        self.handler.stream.truncate(0)
        
        return lines


@search_bp.route('/stream', methods=['POST'])
def search_stream():
    """Stream search process with real-time logs."""
    # Get the search parameters from the request before entering the generator
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
    
    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    top_k = data.get('top_k', 5)
    refine_query = data.get('refine_query', False)
    config = current_app.config['RDB_CONFIG']
    
    def generate():
        try:
            # Start log capture
            log_capture = LogCapture()
            log_capture.start_capture()
            
            yield f"data: {json.dumps({'type': 'log', 'message': '> Starting search request...'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Query: {query}'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Top K: {top_k}, Refine: {refine_query}'})}\n\n"
            
            # Small delay to let initial logs show
            time.sleep(0.1)
            
            # Initialize retriever (this will generate the interesting logs)
            yield f"data: {json.dumps({'type': 'log', 'message': '> Initializing retriever...'})}\n\n"
            
            retriever = DocumentRetriever(config)
            
            # Check for new logs after retriever init
            time.sleep(0.5)  # Give time for logs to be generated
            for log_line in log_capture.get_new_logs():
                yield f"data: {json.dumps({'type': 'log', 'message': f'  {log_line}'})}\n\n"
            
            yield f"data: {json.dumps({'type': 'log', 'message': '> Loading search index...'})}\n\n"
            
            if not retriever.load_index():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Search index not available. Please build index first.'})}\n\n"
                return
            
            # Check for logs after index loading
            time.sleep(0.5)
            for log_line in log_capture.get_new_logs():
                yield f"data: {json.dumps({'type': 'log', 'message': f'  {log_line}'})}\n\n"
            
            # Log search to database
            db_manager = DatabaseManager(config)
            search_data = {
                'original_query': query,
                'top_k': top_k,
                'query_refinement_enabled': refine_query
            }
            
            yield f"data: {json.dumps({'type': 'log', 'message': '> Performing search...'})}\n\n"
            
            # Perform search
            with Timer() as timer:
                results = retriever.search(query, top_k=top_k, refine_query=refine_query)
            
            # Check for logs during search
            time.sleep(0.2)
            for log_line in log_capture.get_new_logs():
                yield f"data: {json.dumps({'type': 'log', 'message': f'  {log_line}'})}\n\n"
            
            # Update search log
            search_data.update({
                'refined_query': results[0]['final_query'] if results else query,
                'results_count': len(results),
                'search_time_ms': int(timer.elapsed * 1000)
            })
            db_manager.log_search(search_data)
            
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Search completed in {timer.elapsed*1000:.0f}ms'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Found {len(results)} results'})}\n\n"
            
            # Stop log capture
            log_capture.stop_capture()
            
            # Send final results
            response_data = {
                'query': query,
                'refined_query': results[0]['final_query'] if results else query,
                'results': results,
                'search_time_ms': search_data['search_time_ms'],
                'total_results': len(results)
            }
            
            yield f"data: {json.dumps({'type': 'results', 'data': response_data})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"Search stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # Ensure log capture is stopped
            if 'log_capture' in locals():
                log_capture.stop_capture()
    
    return Response(generate(), mimetype='text/event-stream')


@search_bp.route('/query', methods=['POST'])
def search_query():
    """Search documents (original endpoint for compatibility)."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        top_k = data.get('top_k', 5)
        refine_query = data.get('refine_query', False)
        
        config = current_app.config['RDB_CONFIG']
        
        # Initialize retriever
        retriever = DocumentRetriever(config)
        if not retriever.load_index():
            return jsonify({'error': 'Search index not available. Please build index first.'}), 503
        
        # Log search
        db_manager = DatabaseManager(config)
        search_data = {
            'original_query': query,
            'top_k': top_k,
            'query_refinement_enabled': refine_query
        }
        
        # Perform search
        with Timer() as timer:
            results = retriever.search(query, top_k=top_k, refine_query=refine_query)
        
        # Update search log
        search_data.update({
            'refined_query': results[0]['final_query'] if results else query,
            'results_count': len(results),
            'search_time_ms': int(timer.elapsed * 1000)
        })
        db_manager.log_search(search_data)
        
        return jsonify({
            'query': query,
            'refined_query': results[0]['final_query'] if results else query,
            'results': results,
            'search_time_ms': search_data['search_time_ms'],
            'total_results': len(results)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/history', methods=['GET'])
def search_history():
    """Get search history."""
    try:
        limit = request.args.get('limit', 20, type=int)
        config = current_app.config['RDB_CONFIG']
        
        db_manager = DatabaseManager(config)
        history = db_manager.get_recent_searches(limit)
        stats = db_manager.get_search_stats()
        
        return jsonify({
            'history': history,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Search history error: {e}")
        return jsonify({'error': str(e)}), 500


@search_bp.route('/suggestions', methods=['GET'])
def search_suggestions():
    """Get search suggestions (simple implementation)."""
    try:
        config = current_app.config['RDB_CONFIG']
        
        # Simple suggestions based on common Arch Wiki topics
        suggestions = [
            "wifi connection problems",
            "install arch linux",
            "graphics driver issues", 
            "sound not working",
            "pacman package manager",
            "systemd services",
            "bluetooth configuration",
            "desktop environment setup",
            "network configuration",
            "boot loader grub"
        ]
        
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        return jsonify({'error': str(e)}), 500
