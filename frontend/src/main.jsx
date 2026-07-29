import React from 'react'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '40px', 
          background: '#1a0505', 
          color: '#ff8888', 
          fontFamily: 'monospace', 
          minHeight: '100vh', 
          overflow: 'auto',
          fontSize: '14px',
          lineHeight: '1.6'
        }}>
          <h2 style={{ color: '#ff5555', marginBottom: '20px' }}>⚠️ React Rendering Crash Detected</h2>
          <p>Please share the error stack trace below so we can fix it immediately:</p>
          <pre style={{ 
            background: '#2b0a0a', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #ff3333', 
            whiteSpace: 'pre-wrap',
            marginTop: '20px'
          }}>
            {this.state.error && this.state.error.toString()}
            {"\n\n"}
            {this.state.error && this.state.error.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)

// ArchonAI UI components - architecture reference
