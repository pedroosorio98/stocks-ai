// Updated chart creation function for script.js

async function createChart() {
    const chartPrompt = document.getElementById('chartPrompt').value;
    const chartDiv = document.getElementById('chart');
    const createButton = event.target;
    
    if (!chartPrompt.trim()) {
        alert('Please enter a chart description (e.g., "Plot NVDA revenue vs stock price")');
        return;
    }
    
    createButton.disabled = true;
    const originalText = createButton.innerHTML;
    createButton.innerHTML = '<span class="spinner"></span> Creating...';
    chartDiv.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div class="spinner" style="margin: 0 auto;"></div>
            <p style="margin-top: 1rem; color: #6c757d;">
                AI is analyzing your request...<br>
                <small>Scanning CSV files, identifying data, generating code...</small>
            </p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/chart', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: chartPrompt })
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            // Render Plotly chart
            Plotly.newPlot('chart', result.data.data, result.data.layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false
            });
            
            // Show generated code in a collapsible section (optional)
            if (result.code) {
                const codeSection = document.createElement('details');
                codeSection.style.marginTop = '1rem';
                codeSection.innerHTML = `
                    <summary style="cursor: pointer; color: #043A22; font-weight: 500;">
                        🔍 View Generated Code
                    </summary>
                    <pre style="background: #f5f5f5; padding: 1rem; border-radius: 8px; overflow-x: auto; margin-top: 0.5rem;"><code>${result.code}</code></pre>
                `;
                chartDiv.appendChild(codeSection);
            }
            
        } else {
            chartDiv.innerHTML = `
                <div style="padding: 1.5rem; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 8px;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #721c24;">❌ Chart Generation Failed</h4>
                    <p style="margin: 0; color: #721c24;"><strong>Error:</strong> ${result.error || 'Unknown error'}</p>
                    ${result.traceback ? `
                        <details style="margin-top: 1rem;">
                            <summary style="cursor: pointer; color: #721c24;">View Error Details</summary>
                            <pre style="background: #fff; padding: 0.5rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; margin-top: 0.5rem;">${result.traceback}</pre>
                        </details>
                    ` : ''}
                    ${result.code ? `
                        <details style="margin-top: 1rem;">
                            <summary style="cursor: pointer; color: #721c24;">View Generated Code</summary>
                            <pre style="background: #fff; padding: 0.5rem; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; margin-top: 0.5rem;">${result.code}</pre>
                        </details>
                    ` : ''}
                </div>
            `;
        }
    } catch (error) {
        chartDiv.innerHTML = `
            <div style="padding: 1.5rem; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 8px;">
                <h4 style="margin: 0 0 0.5rem 0; color: #721c24;">❌ Network Error</h4>
                <p style="margin: 0; color: #721c24;">${error.message}</p>
            </div>
        `;
    } finally {
        createButton.disabled = false;
        createButton.innerHTML = originalText;
    }
}