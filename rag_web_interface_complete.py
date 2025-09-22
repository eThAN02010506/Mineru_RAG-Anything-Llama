#!/usr/bin/env python3
"""
Web\u754c\u9762for RAG-Anything\u7cfb\u7edf
\u63d0\u4f9b\u7b80\u5355\u7684\u7528\u6237\u754c\u9762\u6765\u67e5\u8be2\u6587\u6863\u548c\u67e5\u770b\u56de\u7b54
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# HTML\u6a21\u677f
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG-Anything \u4ea4\u4e92\u754c\u9762</title>
    <style>
        body {
            font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 30px;
            background-color: #f9f9f9;
            border-left: 4px solid #3498db;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
        }
        .sources {
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }
        .source-item {
            margin-bottom: 5px;
        }
        .loading {
            text-align: center;
            margin: 20px 0;
            display: none;
        }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: #3498db;
            animation: spin 1s linear infinite;
            display: inline-block;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status {
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }
        .error {
            color: #e74c3c;
            font-weight: bold;
        }
        .file-info {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #eaf2f8;
            border-radius: 4px;
        }
        .history {
            margin-top: 30px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .history-item {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .history-item:hover {
            background-color: #eaf2f8;
        }
        .history-question {
            font-weight: bold;
            color: #2c3e50;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            border-bottom: 2px solid #3498db;
            color: #3498db;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .system-info {
            margin-top: 20px;
            font-size: 12px;
            color: #999;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>RAG-Anything \u4ea4\u4e92\u754c\u9762</h1>
    <div class="container">
        <div class="tabs">
            <div class="tab active" onclick="switchTab('query')">\u95ee\u7b54</div>
            <div class="tab" onclick="switchTab('history')">\u5386\u53f2\u8bb0\u5f55</div>
            <div class="tab" onclick="switchTab('files')">\u6587\u4ef6\u7ba1\u7406</div>
        </div>
        
        <div id="query-tab" class="tab-content active">
            <div class="form-group">
                <label for="question">\u8bf7\u8f93\u5165\u60a8\u7684\u95ee\u9898:</label>
                <input type="text" id="question" name="question" placeholder="\u4f8b\u5982: \u8fd9\u4e2a\u6587\u6863\u7684\u4e3b\u8981\u5185\u5bb9\u662f\u4ec0\u4e48\uff1f">
            </div>
            
            <button onclick="askQuestion()">\u63d0\u4ea4\u95ee\u9898</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div class="status">\u6b63\u5728\u5904\u7406\u60a8\u7684\u95ee\u9898...</div>
            </div>
            
            <div class="result" id="result" style="display: none;">
                <h3>\u56de\u7b54:</h3>
                <div id="answer"></div>
                
                <div class="sources">
                    <h4>\u6765\u6e90:</h4>
                    <div id="sources"></div>
                </div>
            </div>
        </div>
        
        <div id="history-tab" class="tab-content">
            <h3>\u5386\u53f2\u95ee\u9898</h3>
            <div id="history-list">
                <p>\u6682\u65e0\u5386\u53f2\u8bb0\u5f55</p>
            </div>
        </div>
        
        <div id="files-tab" class="tab-content">
            <h3>\u6587\u6863\u4fe1\u606f</h3>
            <div id="file-list">\u52a0\u8f7d\u4e2d...</div>
            
            <div class="form-group" style="margin-top: 20px;">
                <button onclick="processDocuments()">\u91cd\u65b0\u5904\u7406\u6587\u6863</button>
                <div id="process-status"></div>
            </div>
        </div>
        
        <div class="system-info">
            RAG-Anything Web\u754c\u9762 v1.0
        </div>
    </div>

    <script>
        // \u5b58\u50a8\u5386\u53f2\u8bb0\u5f55
        let questionHistory = [];
        
        // \u9875\u9762\u52a0\u8f7d\u65f6\u83b7\u53d6\u6587\u4ef6\u4fe1\u606f
        window.onload = function() {
            fetchFileInfo();
            loadHistory();
        };
        
        function switchTab(tabName) {
            // \u9690\u85cf\u6240\u6709\u6807\u7b7e\u5185\u5bb9
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // \u53d6\u6d88\u6240\u6709\u6807\u7b7e\u7684\u6d3b\u52a8\u72b6\u6001
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // \u6fc0\u6d3b\u9009\u5b9a\u7684\u6807\u7b7e
            document.getElementById(tabName + '-tab').classList.add('active');
            document.querySelectorAll('.tab').forEach(tab => {
                if (tab.textContent.toLowerCase().includes(tabName)) {
                    tab.classList.add('active');
                }
            });
            
            // \u5982\u679c\u5207\u6362\u5230\u6587\u4ef6\u6807\u7b7e\uff0c\u5237\u65b0\u6587\u4ef6\u4fe1\u606f
            if (tabName === 'files') {
                fetchFileInfo();
            }
        }
        
        function fetchFileInfo() {
            fetch('/files')
                .then(response => response.json())
                .then(data => {
                    const fileListDiv = document.getElementById('file-list');
                    if (data.files && data.files.length > 0) {
                        let html = '<ul>';
                        data.files.forEach(file => {
                            html += `<li>${file}</li>`;
                        });
                        html += '</ul>';
                        
                        if (data.chunks_exists) {
                            html += '<p><strong>\u72b6\u6001:</strong> \u6587\u6863\u5df2\u5904\u7406</p>';
                        } else {
                            html += '<p><strong>\u72b6\u6001:</strong> \u6587\u6863\u5c1a\u672a\u5904\u7406</p>';
                        }
                        
                        fileListDiv.innerHTML = html;
                    } else {
                        fileListDiv.innerHTML = '<p>\u6ca1\u6709\u627e\u5230PDF\u6587\u4ef6\u3002\u8bf7\u5c06PDF\u6587\u4ef6\u653e\u5165data\u76ee\u5f55\u3002</p>';
                    }
                })
                .catch(error => {
                    console.error('Error fetching file info:', error);
                    document.getElementById('file-list').innerHTML = '<p class="error">\u83b7\u53d6\u6587\u4ef6\u4fe1\u606f\u5931\u8d25</p>';
                });
        }
        
        function processDocuments() {
            const statusDiv = document.getElementById('process-status');
            statusDiv.innerHTML = '<p>\u6b63\u5728\u5904\u7406\u6587\u6863\uff0c\u8bf7\u7a0d\u5019...</p>';
            
            fetch('/process', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    statusDiv.innerHTML = '<p>\u6587\u6863\u5904\u7406\u6210\u529f!</p>';
                    fetchFileInfo();
                } else {
                    statusDiv.innerHTML = `<p class="error">\u6587\u6863\u5904\u7406\u5931\u8d25: ${data.error}</p>`;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                statusDiv.innerHTML = '<p class="error">\u5904\u7406\u8bf7\u6c42\u65f6\u51fa\u9519</p>';
            });
        }
        
        function askQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('\u8bf7\u8f93\u5165\u95ee\u9898');
                return;
            }
            
            // \u663e\u793a\u52a0\u8f7d\u52a8\u753b
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            // \u53d1\u9001\u8bf7\u6c42
            fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `question=${encodeURIComponent(question)}`
            })
            .then(response => response.json())
            .then(data => {
                // \u9690\u85cf\u52a0\u8f7d\u52a8\u753b
                document.getElementById('loading').style.display = 'none';
                
                // \u663e\u793a\u7ed3\u679c
                document.getElementById('result').style.display = 'block';
                document.getElementById('answer').textContent = data.answer || '\u6ca1\u6709\u627e\u5230\u56de\u7b54';
                
                // \u663e\u793a\u6765\u6e90
                const sourcesDiv = document.getElementById('sources');
                if (data.sources && data.sources.length > 0) {
                    let html = '';
                    data.sources.forEach((source, index) => {
                        html += `<div class="source-item">\u7247\u6bb5 ${index + 1}: ${source}</div>`;
                    });
                    sourcesDiv.innerHTML = html;
                } else {
                    sourcesDiv.innerHTML = '<p>\u6ca1\u6709\u6765\u6e90\u4fe1\u606f</p>';
                }
                
                // \u6dfb\u52a0\u5230\u5386\u53f2\u8bb0\u5f55
                addToHistory(question, data.answer);
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result').style.display = 'block';
                document.getElementById('answer').innerHTML = '<p class="error">\u5904\u7406\u8bf7\u6c42\u65f6\u51fa\u9519</p>';
                document.getElementById('sources').innerHTML = '';
            });
        }
        
        function addToHistory(question, answer) {
            // \u6dfb\u52a0\u5230\u5386\u53f2\u8bb0\u5f55\u6570\u7ec4
            questionHistory.unshift({
                question: question,
                answer: answer,
                timestamp: new Date().toLocaleString()
            });
            
            // \u6700\u591a\u4fdd\u5b5810\u6761\u8bb0\u5f55
            if (questionHistory.length > 10) {
                questionHistory.pop();
            }
            
            // \u4fdd\u5b58\u5230\u672c\u5730\u5b58\u50a8
            localStorage.setItem('ragQuestionHistory', JSON.stringify(questionHistory));
            
            // \u66f4\u65b0\u5386\u53f2\u8bb0\u5f55\u663e\u793a
            updateHistoryDisplay();
        }
        
        function loadHistory() {
            // \u4ece\u672c\u5730\u5b58\u50a8\u52a0\u8f7d\u5386\u53f2\u8bb0\u5f55
            const savedHistory = localStorage.getItem('ragQuestionHistory');
            if (savedHistory) {
                questionHistory = JSON.parse(savedHistory);
                updateHistoryDisplay();
            }
        }
        
        function updateHistoryDisplay() {
            const historyListDiv = document.getElementById('history-list');
            if (questionHistory.length === 0) {
                historyListDiv.innerHTML = '<p>\u6682\u65e0\u5386\u53f2\u8bb0\u5f55</p>';
                return;
            }
            
            let html = '';
            questionHistory.forEach((item, index) => {
                html += `
                    <div class="history-item" onclick="loadHistoryItem(${index})">
                        <div class="history-question">${item.question}</div>
                        <div class="history-time">${item.timestamp}</div>
                    </div>
                `;
            });
            
            historyListDiv.innerHTML = html;
        }
        
        function loadHistoryItem(index) {
            const item = questionHistory[index];
            if (item) {
                // \u5207\u6362\u5230\u67e5\u8be2\u6807\u7b7e
                switchTab('query');
                
                // \u586b\u5145\u95ee\u9898
                document.getElementById('question').value = item.question;
                
                // \u663e\u793a\u56de\u7b54
                document.getElementById('result').style.display = 'block';
                document.getElementById('answer').textContent = item.answer || '\u6ca1\u6709\u627e\u5230\u56de\u7b54';
                document.getElementById('sources').innerHTML = '<p>\u5386\u53f2\u8bb0\u5f55\u4e2d\u6ca1\u6709\u4fdd\u5b58\u6765\u6e90\u4fe1\u606f</p>';
            }
        }
        
        // \u6309Enter\u952e\u63d0\u4ea4\u95ee\u9898
        document.getElementById('question').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
"""


class RAGServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/files":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()

            # \u83b7\u53d6data\u76ee\u5f55\u4e2d\u7684PDF\u6587\u4ef6
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
            pdf_files = []
            chunks_exists = False

            if os.path.exists(data_dir):
                pdf_files = [
                    f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")
                ]
                chunks_path = os.path.join(data_dir, "chunks.jsonl")
                chunks_exists = os.path.exists(chunks_path)

            response = {"files": pdf_files, "chunks_exists": chunks_exists}

            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/ask":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            params = urllib.parse.parse_qs(post_data)

            question = params.get("question", [""])[0]

            if question:
                # \u8c03\u7528RAG-Anything\u7cfb\u7edf
                answer, sources = self.query_rag_system(question)

                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()

                response = {"answer": answer, "sources": sources}

                self.wfile.write(json.dumps(response).encode("utf-8"))
            else:
                self.send_response(400)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()

                response = {"error": "\u6ca1\u6709\u63d0\u4f9b\u95ee\u9898"}

                self.wfile.write(json.dumps(response).encode("utf-8"))
        elif self.path == "/process":
            # \u5904\u7406\u6587\u6863
            success, error_msg = self.process_documents()

            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.end_headers()

            response = {"success": success, "error": error_msg}

            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def query_rag_system(self, question):
        """\u8c03\u7528RAG-Anything\u7cfb\u7edf\u5e76\u8fd4\u56de\u56de\u7b54"""
        try:
            # \u83b7\u53d6\u5f53\u524d\u811a\u672c\u76ee\u5f55
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")

            # \u68c0\u67e5\u662f\u5426\u5df2\u7ecf\u5904\u7406\u8fc7\u6587\u6863
            chunks_path = os.path.join(data_dir, "chunks.jsonl")
            if not os.path.exists(chunks_path):
                print(
                    "\u6587\u6863\u5c1a\u672a\u5904\u7406\uff0c\u5148\u5904\u7406\u6587\u6863..."
                )
                success, error_msg = self.process_documents()
                if not success:
                    return f"\u5904\u7406\u6587\u6863\u5931\u8d25: {error_msg}", []

            # \u8c03\u7528run_rag.py
            cmd = [sys.executable, "run_rag.py", data_dir, question]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error running RAG system: {result.stderr}")
                return (
                    f"\u5904\u7406\u95ee\u9898\u65f6\u51fa\u9519: {result.stderr}",
                    [],
                )

            # \u89e3\u6790\u8f93\u51fa
            output = result.stdout

            # \u63d0\u53d6\u56de\u7b54\u548c\u6765\u6e90
            answer = ""
            sources = []

            # \u5c1d\u8bd5\u4ece\u8f93\u51fa\u4e2d\u63d0\u53d6\u56de\u7b54
            answer_match = re.search(
                r"\u56de\u7b54:(.*?)(?:\u56de\u7b54\u57fa\u4e8e|$)", output, re.DOTALL
            )
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                answer = output.strip()

            # \u5c1d\u8bd5\u4ece\u8f93\u51fa\u4e2d\u63d0\u53d6\u6765\u6e90\u4fe1\u606f
            source_match = re.search(
                r"\u56de\u7b54\u57fa\u4e8e\s+(\d+)\s+\u4e2a\u6587\u6863\u7247\u6bb5",
                output,
            )
            if source_match:
                num_sources = int(source_match.group(1))
                sources = [
                    f"\u6587\u6863\u7247\u6bb5 {i+1}" for i in range(num_sources)
                ]

            return answer, sources

        except Exception as e:
            print(f"Error: {str(e)}")
            return f"\u5904\u7406\u95ee\u9898\u65f6\u51fa\u9519: {str(e)}", []

    def process_documents(self):
        """\u5904\u7406\u6587\u6863"""
        try:
            # \u83b7\u53d6\u5f53\u524d\u811a\u672c\u76ee\u5f55
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")

            # \u68c0\u67e5\u662f\u5426\u6709PDF\u6587\u4ef6
            pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
            if not pdf_files:
                return False, "data\u76ee\u5f55\u4e2d\u6ca1\u6709PDF\u6587\u4ef6"

            # \u5220\u9664\u73b0\u6709\u7684\u5904\u7406\u7ed3\u679c
            chunks_path = os.path.join(data_dir, "chunks.jsonl")
            if os.path.exists(chunks_path):
                os.remove(chunks_path)

            faiss_path = os.path.join(data_dir, "faiss_index.index")
            if os.path.exists(faiss_path):
                os.remove(faiss_path)

            mapping_path = os.path.join(data_dir, "mapping.json")
            if os.path.exists(mapping_path):
                os.remove(mapping_path)

            # \u8c03\u7528run_rag.py\u5904\u7406\u6587\u6863
            cmd = [sys.executable, "run_rag.py", data_dir, "--reparse"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error processing documents: {result.stderr}")
                return False, result.stderr

            return True, ""

        except Exception as e:
            print(f"Error: {str(e)}")
            return False, str(e)


def run_server(port=8000):
    """\u8fd0\u884cWeb\u670d\u52a1\u5668"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, RAGServer)
    print(f"\u542f\u52a8\u670d\u52a1\u5668\u5728 http://localhost:{port}")
    print("\u6309Ctrl+C\u505c\u6b62\u670d\u52a1\u5668")
    httpd.serve_forever()


def open_browser(port=8000):
    """\u6253\u5f00\u6d4f\u89c8\u5668"""
    # \u7b49\u5f85\u670d\u52a1\u5668\u542f\u52a8
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}")


def main():
    parser = argparse.ArgumentParser(description="RAG-Anything Web\u754c\u9762")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="\u670d\u52a1\u5668\u7aef\u53e3 (\u9ed8\u8ba4: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="\u4e0d\u81ea\u52a8\u6253\u5f00\u6d4f\u89c8\u5668",
    )

    args = parser.parse_args()

    # \u68c0\u67e5data\u76ee\u5f55\u662f\u5426\u5b58\u5728
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        print(f"\u8b66\u544a: data\u76ee\u5f55\u4e0d\u5b58\u5728: {data_dir}")
        print("\u521b\u5efadata\u76ee\u5f55...")
        os.makedirs(data_dir)

    # \u68c0\u67e5\u662f\u5426\u6709PDF\u6587\u4ef6
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("\u8b66\u544a: data\u76ee\u5f55\u4e2d\u6ca1\u6709PDF\u6587\u4ef6")
        print("\u8bf7\u5c06PDF\u6587\u4ef6\u653e\u5165data\u76ee\u5f55")
    else:
        print(
            f"\u627e\u5230 {len(pdf_files)} \u4e2aPDF\u6587\u4ef6: {', '.join(pdf_files)}"
        )

    # \u68c0\u67e5\u662f\u5426\u5df2\u7ecf\u5904\u7406\u8fc7\u6587\u6863
    chunks_path = os.path.join(data_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        print("\u8b66\u544a: \u6ca1\u6709\u627e\u5230chunks.jsonl\u6587\u4ef6")
        print(
            "\u7cfb\u7edf\u5c06\u5728\u7b2c\u4e00\u6b21\u67e5\u8be2\u65f6\u5904\u7406\u6587\u6863"
        )

    # \u542f\u52a8\u6d4f\u89c8\u5668
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,)).start()

    # \u542f\u52a8\u670d\u52a1\u5668
    run_server(args.port)


if __name__ == "__main__":
    main()
