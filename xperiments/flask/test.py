import numpy as np
from flask import Flask, render_template_string
import os

app = Flask(__name__)

@app.route("/")
def index():
  # Loop over all the files in the folder
  results = []
  for _ in range(1000):
    row = np.random.randint(0,100, size=10).tolist()
    results.append(str(row))

    # Render the template with the current progress
    yield render_template_string("<br>".join(results))

  # Render the final results after all files have been processed
  yield render_template_string("<br>".join(results))

if __name__ == "__main__":
    app.run(debug=True)