// Global variables for report tracking
let currentJobId = null;
let statusCheckInterval = null;
let statusCheckAttempts = 0;
const MAX_STATUS_CHECKS = 300; // 10 minutes max (300 checks * 2 seconds = 600 seconds)

// Section switching with active state management
function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Add active class to clicked button
    document.getElementById('btn-' + sectionId).classList.add('active');
}

// AI Prompt submission with timeout
async function submitPrompt() {
    const promptInput = document.getElementById('promptInput');
    const responseDiv = document.getElementById('promptResponse');
    const submitButton = event.target;
    const useWebCheckbox = document.getElementById('useWebSearch');
    
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        responseDiv.innerHTML = '<p style="color: #dc3545; padding: 1rem; background: #f8d7da; border-radius: 8px;">⚠️ Please enter a prompt.</p>';
        return;
    }
    
    // Show loading state
    submitButton.disabled = true;
    const originalText = submitButton.innerHTML;
    submitButton.innerHTML = '<span class="spinner"></span> Processing...';
    responseDiv.innerHTML = `
        <div style="text-align: center; padding: 2rem; color: #6c757d;">
            <div class="spinner" style="margin: 0 auto 1rem;"></div>
            <p>Searching internal documents${useWebCheckbox.checked ? ' and web' : ''}...</p>
        </div>
    `;
    
    try {
        // Add timeout to fetch request (60 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
        
        const response = await fetch('/api/prompt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                prompt: prompt,
                use_web: useWebCheckbox.checked
            }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            const formattedResponse = formatResponse(data.response);
            responseDiv.innerHTML = `
                <div class="ai-response">
                    <h3>AI Response</h3>
                    <div>${formattedResponse}</div>
                </div>
            `;
        } else {
            responseDiv.innerHTML = `
                <div style="padding: 1rem; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 8px;">
                    <strong>Error:</strong> ${data.error || 'Unknown error occurred'}
                </div>
            `;
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            responseDiv.innerHTML = `
                <div style="padding: 1rem; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 8px;">
                    <strong>Timeout:</strong> Request took too long (>60 seconds). The query might be too complex or the system is overloaded. Try a simpler question.
                </div>
            `;
        } else {
            responseDiv.innerHTML = `
                <div style="padding: 1rem; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 8px;">
                    <strong>Network Error:</strong> ${error.message}
                </div>
            `;
        }
    } finally {
        submitButton.disabled = false;
        submitButton.innerHTML = originalText;
    }
}

// Format response with basic markdown support
function formatResponse(text) {
    // Convert **bold**
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert *italic*
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert line breaks
    text = text.replace(/\n/g, '<br>');
    
    // Highlight "Sources:" section
    text = text.replace(/(Sources:)/gi, '<br><br><strong style="color: #043A22; font-size: 1.1rem;">$1</strong><br>');
    
    // Make URLs clickable
    text = text.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank">$1</a>');
    
    return text;
}

// Report generation with async progress tracking and timeout
async function generateReport() {
    const companySelect = document.getElementById('companySelect');
    const downloadLink = document.getElementById('downloadLink');
    const generateButton = event.target;
    
    const company = companySelect.value;
    
    if (!company) {
        alert('Please select a company');
        return;
    }
    
    // Clear any previous status check
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    
    // Reset attempt counter
    statusCheckAttempts = 0;
    
    // Disable button and show loading state
    generateButton.disabled = true;
    const originalText = generateButton.innerHTML;
    generateButton.innerHTML = '<span class="spinner"></span> Running...';
    
    // Show generating state in download link
    downloadLink.style.display = 'inline-flex';
    downloadLink.innerHTML = '<span class="spinner"></span> Starting report generation...';
    downloadLink.style.pointerEvents = 'none';
    downloadLink.removeAttribute('href');
    
    try {
        // Start report generation
        const response = await fetch('/api/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ company: company })
        });
        
        const data = await response.json();
        
        if (response.ok && data.status === 'success') {
            currentJobId = data.job_id;
            
            // Show initial status
            downloadLink.innerHTML = `
                <span class="spinner"></span> Generating report for ${company}...
            `;
            
            // Start polling for status every 2 seconds
            statusCheckInterval = setInterval(() => checkReportStatus(currentJobId), 2000);
            
        } else {
            downloadLink.innerHTML = `
                <span style="color: #dc3545;">❌</span> Error: ${data.error || 'Failed to start generation'}
            `;
            downloadLink.style.pointerEvents = 'auto';
            generateButton.disabled = false;
            generateButton.innerHTML = originalText;
        }
        
    } catch (error) {
        downloadLink.innerHTML = `
            <span style="color: #dc3545;">❌</span> Network error: ${error.message}
        `;
        downloadLink.style.pointerEvents = 'auto';
        generateButton.disabled = false;
        generateButton.innerHTML = originalText;
    }
}

// Check report generation status with timeout protection
async function checkReportStatus(jobId) {
    statusCheckAttempts++;
    
    // TIMEOUT: Stop after MAX_STATUS_CHECKS attempts
    if (statusCheckAttempts > MAX_STATUS_CHECKS) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        
        const downloadLink = document.getElementById('downloadLink');
        const generateButton = document.querySelector('#aiReportGenerator .primary-btn');
        
        downloadLink.innerHTML = `
            <span style="color: #dc3545;">⏱️</span> Report generation timed out after 10 minutes. 
            <br>The report may still be processing. Please check back later or try again.
        `;
        downloadLink.style.pointerEvents = 'auto';
        downloadLink.style.background = '#dc3545';
        
        // Re-enable generate button
        if (generateButton) {
            generateButton.disabled = false;
            generateButton.innerHTML = '<span class="btn-icon"></span> Generate Report';
        }
        
        console.error(`Report generation timeout after ${statusCheckAttempts} attempts`);
        return;
    }
    
    try {
        const response = await fetch(`/api/report/status/${jobId}`);
        const data = await response.json();
        
        const downloadLink = document.getElementById('downloadLink');
        const generateButton = document.querySelector('#aiReportGenerator .primary-btn');
        
        if (data.status === 'completed') {
            // Stop polling
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
            
            // Show download link
            downloadLink.href = data.download_url;
            downloadLink.innerHTML = `
                <span class="btn-icon">📄</span> Download ${data.filename}
            `;
            downloadLink.style.pointerEvents = 'auto';
            downloadLink.style.background = 'var(--success-color)';
            
            // Re-enable generate button
            if (generateButton) {
                generateButton.disabled = false;
                generateButton.innerHTML = '<span class="btn-icon"></span> Generate Report';
            }
            
            console.log(`Report completed after ${statusCheckAttempts} checks`);
            
        } else if (data.status === 'failed') {
            // Stop polling
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
            
            // Show error
            downloadLink.innerHTML = `
                <span style="color: #dc3545;">❌</span> Generation failed: ${data.error || 'Unknown error'}
            `;
            downloadLink.style.pointerEvents = 'auto';
            downloadLink.style.background = '#dc3545';
            
            // Re-enable generate button
            if (generateButton) {
                generateButton.disabled = false;
                generateButton.innerHTML = '<span class="btn-icon"></span> Generate Report';
            }
            
        } else if (data.status === 'generating') {
            // Still generating - update status with progress indicator
            const elapsed = Math.floor((statusCheckAttempts * 2) / 60); // Minutes elapsed
            downloadLink.innerHTML = `
                <span class="spinner"></span> Generating ${data.company} report (${data.ticker})...
            `;
        } else if (data.status === 'pending') {
            // Just started
            downloadLink.innerHTML = `
                <span class="spinner"></span> Initializing report for ${data.company}...
            `;
        }
        
    } catch (error) {
        console.error('Error checking status:', error);
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
        
        const downloadLink = document.getElementById('downloadLink');
        const generateButton = document.querySelector('#aiReportGenerator .primary-btn');
        
        downloadLink.innerHTML = `
            <span style="color: #dc3545;">❌</span> Error checking status: ${error.message}
        `;
        downloadLink.style.pointerEvents = 'auto';
        
        // Re-enable generate button
        if (generateButton) {
            generateButton.disabled = false;
            generateButton.innerHTML = '<span class="btn-icon"></span> Generate Report';
        }
    }
}

// Chart creation
async function createChart() {
    const chartPrompt = document.getElementById('chartPrompt').value;
    const chartDiv = document.getElementById('chart');
    const createButton = event.target;
    
    if (!chartPrompt.trim()) {
        alert('Please enter chart data');
        return;
    }
    
    createButton.disabled = true;
    const originalText = createButton.innerHTML;
    createButton.innerHTML = '<span class="spinner"></span> Creating...';
    chartDiv.innerHTML = '<div style="text-align: center; padding: 2rem;"><div class="spinner" style="margin: 0 auto;"></div><p style="margin-top: 1rem; color: #6c757d;">Creating chart...</p></div>';
    
    try {
        const response = await fetch('/api/chart', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ data: chartPrompt })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            Plotly.newPlot('chart', data.data, {
                responsive: true,
                displayModeBar: true
            });
        } else {
            chartDiv.innerHTML = '<p style="color: #dc3545; padding: 1rem;">Error creating chart</p>';
        }
    } catch (error) {
        chartDiv.innerHTML = `<p style="color: #dc3545; padding: 1rem;">Network error: ${error.message}</p>`;
    } finally {
        createButton.disabled = false;
        createButton.innerHTML = originalText;
    }
}

// Allow Enter key to submit prompt (Shift+Enter for new line)
document.addEventListener('DOMContentLoaded', function() {
    const promptInput = document.getElementById('promptInput');
    
    if (promptInput) {
        promptInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submitPrompt();
            }
        });
    }
});