"""Find the distributions given the clustering."""
import os
import pdb


def find_dist(path, optional=False):
    """Find the distributions."""
    dist = {}
    clusters = os.listdir(path)
    n_clusters = len(clusters)
    for i, c in enumerate(clusters):
        print("Cluster", i)
        cluster_path = path + c + '/'
        images = os.listdir(cluster_path)
        for img in images:
            name = img.split('-')[0] + '-' + img.split('-')[1]
            if optional:
                name = img.split('-')[0]
            if name not in dist:
                dist[name] = [0.0 for j in range(n_clusters)]
            dist[name][i] += 1.0
    for img in dist.keys():
        dist[img] = [float(x)-min(dist[img])/max(dist[img]) for x in dist[img]]
    return dist


if __name__ == "__main__":
    PATH = "/home/chrizandr/data/Bangla/strokes_linear_23/"
    dist = find_dist(PATH, False)
    f = open("strokes_linear_23ban.csv", "w")
    for key in dist.keys():
        f.write(",".join([str(x) for x in dist[key]]))
        f.write(',' + key + '\n')
    print("Done")
    f.close()
