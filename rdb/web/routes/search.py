"""
Search API routes using subprocess approach.
"""

import json
import subprocess
import time
from flask import Blueprint, request, jsonify, current_app, Response
from rdb.storage.database import DatabaseManager
from rdb.utils.helpers import Timer

search_bp = Blueprint('search', __name__)


@search_bp.route('/stream', methods=['POST'])
def search_stream():
    """Stream search process using subprocess."""
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
            yield f"data: {json.dumps({'type': 'log', 'message': '> Starting fresh search process...'})}\n\n"
            time.sleep(0.2)
            
            # Build CLI command
            cmd = ['rdb', 'search', query, '--top-k', str(top_k)]
            if refine_query:
                cmd.append('--refine')
            else:
                cmd.append('--no-refine')
            
            cmd_str = " ".join(cmd)
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Running: {cmd_str}'})}\n\n"
            time.sleep(0.2)
            
            yield f"data: {json.dumps({'type': 'log', 'message': '> Subprocess loading models...'})}\n\n"
            
            # Run search in subprocess
            with Timer() as timer:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
            
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error"
                yield f"data: {json.dumps({'type': 'error', 'message': f'Search failed: {error_msg}'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'log', 'message': f'> Subprocess completed in {timer.elapsed*1000:.0f}ms'})}\n\n"
            yield f"data: {json.dumps({'type': 'log', 'message': '> Process terminated, all memory freed'})}\n\n"
            
            # Safely encode CLI output to avoid JSON parsing issues
            import base64
            cli_output_encoded = base64.b64encode(result.stdout.encode('utf-8')).decode('ascii')

            response_data = {
                'query': query,
                'refined_query': query,
                'results': [],
                'search_time_ms': int(timer.elapsed * 1000),
                'total_results': 0,
                'cli_output_encoded': cli_output_encoded
            }
            
            yield f"data: {json.dumps({'type': 'results', 'data': response_data})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
            
        except subprocess.TimeoutExpired:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Search timed out after 2 minutes'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@search_bp.route('/query', methods=['POST'])
def search_query():
    """Simple search using subprocess."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        top_k = data.get('top_k', 5)
        refine_query = data.get('refine_query', False)
        
        # Build CLI command
        cmd = ['rdb', 'search', query, '--top-k', str(top_k)]
        if refine_query:
            cmd.append('--refine')
        else:
            cmd.append('--no-refine')
        
        # Run search in subprocess
        with Timer() as timer:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            return jsonify({'error': f'Search failed: {error_msg}'}), 500
        
        # Log search to database
        db_manager = DatabaseManager(current_app.config['RDB_CONFIG'])
        search_data = {
            'original_query': query,
            'refined_query': query,
            'top_k': top_k,
            'query_refinement_enabled': refine_query,
            'results_count': 0,  # CLI doesn't easily return count
            'search_time_ms': int(timer.elapsed * 1000)
        }
        db_manager.log_search(search_data)
        
        return jsonify({
            'query': query,
            'refined_query': query,
            'results': [],
            'search_time_ms': int(timer.elapsed * 1000),
            'total_results': 0,
            'cli_output': result.stdout
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Search timed out after 2 minutes'}), 500
    except Exception as e:
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
        return jsonify({'error': str(e)}), 500


@search_bp.route('/suggestions', methods=['GET'])
def search_suggestions():
    """Get search suggestions."""
    try:
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
        return jsonify({'error': str(e)}), 500
