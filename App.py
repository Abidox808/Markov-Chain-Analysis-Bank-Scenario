from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Setting non-interactive backend
import matplotlib.pyplot as plt
import time


app = Flask(__name__)

# Default transition probability matrix and initial state
P = np.array([[0.8, 0.1, 0.1],
              [0.2, 0.6, 0.2],
              [0.1, 0.2, 0.7]])

initial_state = np.array([[1, 0, 0]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    global P, initial_state
    steps = int(request.form['steps'])
    # Get form inputs for transition probabilities
    P = np.array([[float(request.form['p11']), float(request.form['p12']), float(request.form['p13'])],
                  [float(request.form['p21']), float(request.form['p22']), float(request.form['p23'])],
                  [float(request.form['p31']), float(request.form['p32']), float(request.form['p33'])]])

    # Update the global initial_state
    initial_state = np.array([[float(request.form['s1']), float(request.form['s2']), float(request.form['s3'])]])

    # Simulate Markov chain using a copy of the global initial_state
    state = np.copy(initial_state)
    stateHist = np.copy(initial_state)

    for _ in range(steps):
        state = np.dot(state, P)
        stateHist = np.vstack([stateHist, state])  
    
    dfDistrHist = pd.DataFrame(stateHist, columns=['BMCE', 'ATTIJARIWAFA Bank', 'CIH'])
    plot_path, heatmap_path, histogram_path, pie_path = generate_plots(request.form, dfDistrHist)

    
    return render_template('result.html', plot_path=plot_path ,heatmap_path=heatmap_path,histogram_path=histogram_path,pie_path=pie_path)

def generate_plots(form_data,dfDistrHist):
    plt.figure(figsize=(10, 6))
    
    for company in dfDistrHist.columns:
        sns.lineplot(data=dfDistrHist[company], label=company)
    
    plt.xlabel('Pas de temps')
    plt.ylabel('Pourcentage de fidélité')
    plt.title('Évolution de la fidélité des clients au fil du temps')

    # Set y-axis limits to range from 0 to 1
    plt.ylim(0, 1)

    # Save the plot as a file
    plot_path = 'static/loyalty_evolution.png'
    plt.savefig(plot_path)
    plt.close()
    
     # Create a heatmap of the transition matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(P, annot=True, cmap="YlGnBu", xticklabels=['BMCE', 'ATTIJARIWAFA Bank', 'CIH'], yticklabels=['BMCE', 'ATTIJARIWAFA Bank', 'CIH'])
    plt.xlabel('à')
    plt.ylabel('De')
    plt.title('Probabilités de transition entre Les 3 Banques')
    heatmap_path = 'static/transition_heatmap.png'
    plt.savefig(heatmap_path)
    plt.close()


    # Histogram for loyalty percentages across all time steps for each company
    plt.figure(figsize=(8, 6))
    for company in dfDistrHist.columns:
        sns.histplot(dfDistrHist[company], label=company, kde=True, bins=20)  # Adjust bins as needed
    plt.xlabel('Pourcentage de fidélité')
    plt.ylabel('Fréquence')
    plt.title('Répartition des pourcentages de Fidélité')
    plt.legend()
    histogram_path = 'static/loyalty_histogram.png'
    plt.savefig(histogram_path)
    plt.close()

    # Pie chart for loyalty percentages at the last time step
    last_time_step = dfDistrHist.index[-1]  # Getting the last time step
    loyalty_percentages = dfDistrHist.iloc[last_time_step]
    plt.figure(figsize=(8, 6))
    plt.pie(loyalty_percentages, labels=loyalty_percentages.index, autopct='%1.1f%%', startangle=140)
    plt.title('Pourcentages de Fidélité au dernier pas de temps')
    pie_chart_path = 'static/loyalty_pie_chart.png'
    plt.savefig(pie_chart_path)
    plt.close()

    timestamp = int(time.time())  # Generate a timestamp
    plot_path = f'static/loyalty_evolution.png?{timestamp}'
    heatmap_path = f'static/transition_heatmap.png?{timestamp}'
    histogram_path = f'static/loyalty_histogram.png?{timestamp}'
    pie_chart_path = f'static/loyalty_pie_chart.png?{timestamp}'

    return plot_path, heatmap_path, histogram_path ,pie_chart_path

if __name__ == '__main__':
    app.run(debug=True)
