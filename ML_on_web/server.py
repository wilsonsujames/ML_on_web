from flask import Flask,render_template,jsonify,Response

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib

matplotlib.use('agg')
import io
from joblib import dump, load
import pandas as pd
from matplotlib.figure import Figure

app = Flask(__name__)



@app.route('/',methods=[ "GET",'POST'])
def index():
    print('dashboard')

    return render_template('dashboard.html')    



@app.route('/plot')
def plot_png():

    dataset = pd.read_csv('./Mall_Customers.csv')
    # X= dataset.iloc[151:202, [3,4]].values
    X= dataset.iloc[:, [3,4]].values
    kmean_clf = load('kmean.joblib') 
    y_kmeans= kmean_clf.fit_predict(X)

    Kmeansfig = Figure()
    axis = Kmeansfig.add_subplot(1, 1, 1)
    axis.grid(color='lightgrey')
    axis.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    axis.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    axis.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    axis.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    axis.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    axis.scatter(kmean_clf.cluster_centers_[:, 0], kmean_clf.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    axis.set_title("K-means demo")
    axis.legend()

    output = io.BytesIO()
    FigureCanvas(Kmeansfig).print_png(output)
    
    return Response(output.getvalue(), mimetype="image/png")






if __name__ == "__main__":
    app.run(debug=True,threaded=True,port=5566)    



