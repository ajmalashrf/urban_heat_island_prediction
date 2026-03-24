const form = document.getElementById('prediction-form');
const resultBox = document.getElementById('result');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(form);

  const payload = Object.fromEntries(
    [...formData.entries()].map(([key, value]) => [key, Number(value)])
  );

  resultBox.classList.add('hidden');
  resultBox.textContent = 'Predicting...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction request failed.');
    }

    resultBox.innerHTML = `
      <h3>Prediction Result</h3>
      <p><strong>Heat Risk Level:</strong> ${data.label} (Class ${data.prediction})</p>
      <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
    `;
    resultBox.classList.remove('hidden');
  } catch (error) {
    resultBox.innerHTML = `<p><strong>Error:</strong> ${error.message}</p>`;
    resultBox.classList.remove('hidden');
  }
});
