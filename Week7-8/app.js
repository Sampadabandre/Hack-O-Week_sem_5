document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // Tab Navigation
    // -------------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = document.getElementById(`tab-${btn.dataset.tab}`);
            if (targetTab) targetTab.classList.add('active');
        });
    });

    // -------------------------------------------------------------
    // 1. REGRESSION SANDBOX
    // -------------------------------------------------------------
    const polyDegreeInput = document.getElementById('poly-degree');
    const degreeValLabel = document.getElementById('degree-val');
    const regTypeSelect = document.getElementById('reg-type');
    const regAlphaInput = document.getElementById('reg-alpha');
    const alphaValLabel = document.getElementById('alpha-val');
    const alphaGroup = document.getElementById('alpha-group');
    const noiseInput = document.getElementById('noise-level');
    const noiseValLabel = document.getElementById('noise-val');
    const btnResampleReg = document.getElementById('btn-resample-reg');

    let regDataPoints = [];
    let regChart = null;

    function generateRegData() {
        regDataPoints = [];
        const n = 60;
        const noise = parseFloat(noiseInput.value);

        for (let i = 0; i < n; i++) {
            const x = (Math.random() * 6) - 3; // range -3 to 3
            // true curve: 0.5 * x^3 - 0.8 * x^2 - 2 * x + 3
            const trueY = 0.5 * Math.pow(x, 3) - 0.8 * Math.pow(x, 2) - 2 * x + 3;
            const y = trueY + (Math.random() - 0.5) * (noise * 1.5);
            regDataPoints.push({ x, y });
        }
        regDataPoints.sort((a, b) => a.x - b.x);
    }

    function fitPolynomialRegression(points, degree, regType, alpha) {
        // Polynomial feature matrix X
        const N = points.length;
        const X = [];
        const Y = points.map(p => p.y);

        for (let i = 0; i < N; i++) {
            const row = [];
            for (let d = 0; d <= degree; d++) {
                row.push(Math.pow(points[i].x, d));
            }
            X.push(row);
        }

        // Closed form (X^T X + alpha*I)^-1 X^T Y for Ridge, simple gradient update for OLS/Lasso
        const numFeatures = degree + 1;
        let weights = new Array(numFeatures).fill(0);

        // Simple Ridge/OLS via Gradient Descent for UI responsiveness
        const lr = 0.0005;
        const iterations = 8000;

        for (let iter = 0; iter < iterations; iter++) {
            let gradients = new Array(numFeatures).fill(0);

            for (let i = 0; i < N; i++) {
                let pred = 0;
                for (let d = 0; d < numFeatures; d++) {
                    pred += weights[d] * X[i][d];
                }
                const error = pred - Y[i];

                for (let d = 0; d < numFeatures; d++) {
                    gradients[d] += (error * X[i][d]) / N;
                }
            }

            for (let d = 0; d < numFeatures; d++) {
                if (d > 0 && regType === 'ridge') {
                    gradients[d] += (alpha * weights[d]) / N;
                } else if (d > 0 && regType === 'lasso') {
                    gradients[d] += (alpha * Math.sign(weights[d])) / N;
                }
                weights[d] -= lr * gradients[d];
            }
        }

        return weights;
    }

    function updateRegressionPlot() {
        const degree = parseInt(polyDegreeInput.value);
        const regType = regTypeSelect.value;
        const alpha = parseFloat(regAlphaInput.value);

        degreeValLabel.textContent = `Degree ${degree}`;
        alphaValLabel.textContent = alpha.toFixed(1);

        if (regType === 'none') {
            alphaGroup.style.opacity = '0.4';
            regAlphaInput.disabled = true;
        } else {
            alphaGroup.style.opacity = '1.0';
            regAlphaInput.disabled = false;
        }

        const weights = fitPolynomialRegression(regDataPoints, degree, regType, alpha);

        // Predict curve points
        const curvePoints = [];
        const xGrid = [];
        for (let x = -3.2; x <= 3.2; x += 0.1) {
            let yPred = 0;
            for (let d = 0; d <= degree; d++) {
                yPred += weights[d] * Math.pow(x, d);
            }
            curvePoints.push({ x, y: yPred });
        }

        // Calculate Metrics
        let sse = 0, sae = 0, sumY = 0;
        const N = regDataPoints.length;
        regDataPoints.forEach(p => {
            let pred = 0;
            for (let d = 0; d <= degree; d++) {
                pred += weights[d] * Math.pow(p.x, d);
            }
            const err = p.y - pred;
            sse += err * err;
            sae += Math.abs(err);
            sumY += p.y;
        });

        const meanY = sumY / N;
        let sst = 0;
        regDataPoints.forEach(p => { sst += Math.pow(p.y - meanY, 2); });

        const mse = sse / N;
        const rmse = Math.sqrt(mse);
        const mae = sae / N;
        const r2 = Math.max(0, 1 - (sse / sst));

        document.getElementById('reg-mse').textContent = mse.toFixed(2);
        document.getElementById('reg-rmse').textContent = rmse.toFixed(2);
        document.getElementById('reg-mae').textContent = mae.toFixed(2);
        document.getElementById('reg-r2').textContent = r2.toFixed(3);

        // Update Chart
        const ctx = document.getElementById('reg-canvas').getContext('2d');
        if (regChart) regChart.destroy();

        regChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Dataset Points',
                        data: regDataPoints,
                        backgroundColor: '#6366f1',
                        pointRadius: 5
                    },
                    {
                        label: `Fit Curve (${regType.toUpperCase()})`,
                        data: curvePoints,
                        type: 'line',
                        borderColor: regType === 'lasso' ? '#ec4899' : (regType === 'ridge' ? '#14b8a6' : '#f59e0b'),
                        borderWidth: 3,
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                }
            }
        });
    }

    generateRegData();
    updateRegressionPlot();

    polyDegreeInput.addEventListener('input', updateRegressionPlot);
    regTypeSelect.addEventListener('change', updateRegressionPlot);
    regAlphaInput.addEventListener('input', updateRegressionPlot);
    noiseInput.addEventListener('input', () => {
        const val = parseInt(noiseInput.value);
        noiseValLabel.textContent = val < 5 ? 'Low' : (val < 10 ? 'Medium' : 'High');
        generateRegData();
        updateRegressionPlot();
    });
    btnResampleReg.addEventListener('click', () => {
        generateRegData();
        updateRegressionPlot();
    });

    // -------------------------------------------------------------
    // 2. CLASSIFICATION SANDBOX
    // -------------------------------------------------------------
    const clfAlgoSelect = document.getElementById('clf-algorithm');
    const knnkGroup = document.getElementById('knn-k-group');
    const knnKInput = document.getElementById('knn-k');
    const kValLabel = document.getElementById('k-val');
    const thresholdGroup = document.getElementById('threshold-group');
    const thresholdInput = document.getElementById('log-threshold');
    const thresholdValLabel = document.getElementById('threshold-val');
    const datasetShapeSelect = document.getElementById('dataset-shape');
    const btnResampleClf = document.getElementById('btn-resample-clf');
    const clfCanvas = document.getElementById('clf-canvas');
    const clfCtx = clfCanvas.getContext('2d');

    let clfPoints = [];

    function generateClfData() {
        clfPoints = [];
        const type = datasetShapeSelect.value;
        const n = 120;

        for (let i = 0; i < n; i++) {
            if (type === 'moons') {
                const isClass1 = i >= n / 2;
                const angle = Math.random() * Math.PI;
                let x, y;
                if (!isClass1) {
                    x = Math.cos(angle) + (Math.random() - 0.5) * 0.4;
                    y = Math.sin(angle) + (Math.random() - 0.5) * 0.4;
                } else {
                    x = 1 - Math.cos(angle) + (Math.random() - 0.5) * 0.4;
                    y = 0.5 - Math.sin(angle) + (Math.random() - 0.5) * 0.4;
                }
                clfPoints.push({ x, y, label: isClass1 ? 1 : 0 });
            } else {
                // blobs
                const isClass1 = i >= n / 2;
                const cx = isClass1 ? 1.5 : -1.5;
                const cy = isClass1 ? 1.5 : -1.5;
                const x = cx + (Math.random() - 0.5) * 2.2;
                const y = cy + (Math.random() - 0.5) * 2.2;
                clfPoints.push({ x, y, label: isClass1 ? 1 : 0 });
            }
        }
    }

    function renderClassificationCanvas() {
        const algo = clfAlgoSelect.value;
        const k = parseInt(knnKInput.value);
        const threshold = parseFloat(thresholdInput.value);

        kValLabel.textContent = `K = ${k}`;
        thresholdValLabel.textContent = threshold.toFixed(2);

        if (algo === 'knn') {
            knnkGroup.style.display = 'block';
            thresholdGroup.style.display = 'none';
        } else {
            knnkGroup.style.display = 'none';
            thresholdGroup.style.display = 'block';
        }

        const width = clfCanvas.width;
        const height = clfCanvas.height;
        clfCtx.clearRect(0, 0, width, height);

        // Coordinate transforms
        const mapX = (x) => ((x + 3) / 6) * width;
        const mapY = (y) => height - (((y + 3) / 6) * height);
        const unmapX = (px) => (px / width) * 6 - 3;
        const unmapY = (py) => ((height - py) / height) * 6 - 3;

        // Decision boundary grid background
        const imgData = clfCtx.createImageData(width, height);
        const data = imgData.data;

        // Train Logistic Regression model weights simple SGD
        let w0 = 0, w1 = 0, w2 = 0;
        if (algo === 'logistic') {
            for (let iter = 0; iter < 1000; iter++) {
                clfPoints.forEach(p => {
                    const z = w0 + w1 * p.x + w2 * p.y;
                    const prob = 1 / (1 + Math.exp(-z));
                    const err = p.label - prob;
                    w0 += 0.02 * err;
                    w1 += 0.02 * err * p.x;
                    w2 += 0.02 * err * p.y;
                });
            }
        }

        for (let py = 0; py < height; py += 4) {
            for (let px = 0; px < width; px += 4) {
                const x = unmapX(px);
                const y = unmapY(py);
                let predLabel = 0;

                if (algo === 'knn') {
                    // KNN prediction
                    const dists = clfPoints.map(p => ({
                        d: Math.hypot(p.x - x, p.y - y),
                        label: p.label
                    }));
                    dists.sort((a, b) => a.d - b.d);
                    const kNearest = dists.slice(0, k);
                    const count1 = kNearest.filter(p => p.label === 1).length;
                    predLabel = count1 > (k / 2) ? 1 : 0;
                } else {
                    // Logistic Regression
                    const z = w0 + w1 * x + w2 * y;
                    const prob = 1 / (1 + Math.exp(-z));
                    predLabel = prob >= threshold ? 1 : 0;
                }

                // Fill 4x4 block
                const color = predLabel === 1 ? [236, 72, 153, 70] : [99, 102, 241, 70];
                for (let dy = 0; dy < 4; dy++) {
                    for (let dx = 0; dx < 4; dx++) {
                        if (px + dx < width && py + dy < height) {
                            const idx = ((py + dy) * width + (px + dx)) * 4;
                            data[idx] = color[0];
                            data[idx + 1] = color[1];
                            data[idx + 2] = color[2];
                            data[idx + 3] = color[3];
                        }
                    }
                }
            }
        }
        clfCtx.putImageData(imgData, 0, 0);

        // Draw Data Points
        let tp = 0, fp = 0, tn = 0, fn = 0;
        clfPoints.forEach(p => {
            let pred = 0;
            if (algo === 'knn') {
                const dists = clfPoints.map(other => ({
                    d: Math.hypot(other.x - p.x, other.y - p.y),
                    label: other.label
                }));
                dists.sort((a, b) => a.d - b.d);
                const kNearest = dists.slice(0, k);
                const count1 = kNearest.filter(o => o.label === 1).length;
                pred = count1 > (k / 2) ? 1 : 0;
            } else {
                const z = w0 + w1 * p.x + w2 * p.y;
                const prob = 1 / (1 + Math.exp(-z));
                pred = prob >= threshold ? 1 : 0;
            }

            if (p.label === 1 && pred === 1) tp++;
            if (p.label === 0 && pred === 1) fp++;
            if (p.label === 0 && pred === 0) tn++;
            if (p.label === 1 && pred === 0) fn++;

            clfCtx.beginPath();
            clfCtx.arc(mapX(p.x), mapY(p.y), 6, 0, 2 * Math.PI);
            clfCtx.fillStyle = p.label === 1 ? '#ec4899' : '#6366f1';
            clfCtx.fill();
            clfCtx.lineWidth = 1.5;
            clfCtx.strokeStyle = '#ffffff';
            clfCtx.stroke();
        });

        // Compute metrics
        const total = clfPoints.length;
        const acc = ((tp + tn) / total) * 100;
        const prec = (tp + fp) > 0 ? (tp / (tp + fp)) * 100 : 0;
        const rec = (tp + fn) > 0 ? (tp / (tp + fn)) * 100 : 0;
        const f1 = (prec + rec) > 0 ? (2 * prec * rec / (prec + rec)) : 0;

        document.getElementById('clf-acc').textContent = `${acc.toFixed(1)}%`;
        document.getElementById('clf-prec').textContent = `${prec.toFixed(1)}%`;
        document.getElementById('clf-rec').textContent = `${rec.toFixed(1)}%`;
        document.getElementById('clf-f1').textContent = `${f1.toFixed(1)}%`;
    }

    generateClfData();
    renderClassificationCanvas();

    clfAlgoSelect.addEventListener('change', renderClassificationCanvas);
    knnKInput.addEventListener('input', renderClassificationCanvas);
    thresholdInput.addEventListener('input', renderClassificationCanvas);
    datasetShapeSelect.addEventListener('change', () => {
        generateClfData();
        renderClassificationCanvas();
    });
    btnResampleClf.addEventListener('click', () => {
        generateClfData();
        renderClassificationCanvas();
    });

    // -------------------------------------------------------------
    // 3. BENCHMARK METRIC TABLES JSON LOADER
    // -------------------------------------------------------------
    fetch('model_results.json')
        .then(res => res.json())
        .then(data => {
            // Render Regression Table
            let regHtml = `<table><thead><tr>
                <th>Model</th><th>Train RMSE</th><th>Test RMSE</th><th>Train R²</th><th>Test R²</th>
            </tr></thead><tbody>`;
            data.regression.forEach(row => {
                regHtml += `<tr>
                    <td><strong>${row.Model}</strong></td>
                    <td>${row["Train RMSE"]}</td>
                    <td>${row["Test RMSE"]}</td>
                    <td>${row["Train R2"]}</td>
                    <td>${row["Test R2"]}</td>
                </tr>`;
            });
            regHtml += `</tbody></table>`;
            document.getElementById('reg-table-container').innerHTML = regHtml;

            // Render Classification Table
            let clfHtml = `<table><thead><tr>
                <th>Classifier</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th>
            </tr></thead><tbody>`;
            data.classification.forEach(row => {
                clfHtml += `<tr>
                    <td><strong>${row.Classifier}</strong></td>
                    <td>${row.Accuracy}</td>
                    <td>${row.Precision}</td>
                    <td>${row.Recall}</td>
                    <td>${row["F1-Score"]}</td>
                    <td>${row["ROC-AUC"]}</td>
                </tr>`;
            });
            clfHtml += `</tbody></table>`;
            document.getElementById('clf-table-container').innerHTML = clfHtml;
        })
        .catch(err => console.log('JSON benchmark load notice:', err));
});
