import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def main():
    # Here, we are loading the cleaned modelling dataset created in Part A
    df= pd.read_csv("data/modelling_table.csv")

    # Here, we are grouping shipments by hub and calculating their operating statistics
    hub_profile_df = df.groupby("hub_id").agg(
        shipments = ("hub_id", "size"),
        mean_delivery_hours = ("delivery_hours","mean"),
        delay_rate = ("is_delayed", "mean"),
        mean_distance_km = ("distance_km", "mean"),
        mean_n_stops = ("n_stops","mean")
    ).reset_index()

    # Here, we are sorting the hubs by hub_id to maintain a fixed order for K-means
    hub_profile_df = hub_profile_df.sort_values("hub_id").reset_index(drop = True)

    # Here, we are saving the hub profile with the required column names and order
    hub_profile_df.to_csv("outputs/hub_profile.csv", index = False)

    # Here, we are checking the number of hubs and shipments
    print("The total number of hubs is: ", len(hub_profile_df))
    print("The total number of shipment is: ", hub_profile_df["shipments"].sum())
    print("The shipments per hub are: ", hub_profile_df["shipments"].unique())

    # Here, we are displaying the hub profile results
    print("Part D Hub Profile: ")
    print(hub_profile_df.to_string(index = False))

    # Here, we are selecting the four hub statistics used for K-Mean clustering
    profile_features = ["mean_delivery_hours","delay_rate","mean_distance_km","mean_n_stops"]

    # Here, we are extracting the four features from the sorted hubs profile
    X_hubs = hub_profile_df[profile_features]

    # Here, we are scaling the features
    sc = StandardScaler()
    X_hubs_scaled = sc.fit_transform(X_hubs)

    # Here, we are creating three clusters as per instructed in the project
    kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)

    # Here, we are assigning each of the 24 hubs to a cluster
    cluster_labels = kmeans.fit_predict(X_hubs_scaled)

    # Here, we are checking the clusters assigned to each hub
    hub_clusters_df = pd.DataFrame({
        "hub_id": hub_profile_df["hub_id"],
        "cluster": cluster_labels
    })

    print("Part D K-Means Clustering Results: ")
    print(hub_clusters_df.to_string(index = False))

    # Here, we are checking how many hubs belong to each cluster
    print("The number of hubs in each cluster")
    print(hub_clusters_df["cluster"].value_counts().sort_index())

    # Here, we are defining the cluster sizes to evaluate
    k_values = [2,3,4,5]

    # Here, we are calculating the Euclidean distances between all 24 hubs using the same scaled features previously
    distance_matrix = pairwise_distances(X_hubs_scaled, metric = "euclidean")

    # Here, we are storing the separation results for each K
    separation_results = []

    # Here, we are evaluating K-Means with different numbers of clusters
    for k in k_values:
        
        # Here, we are fitting K-Means using the required fixed settings
        kmeans_k = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels_k = kmeans_k.fit_predict(X_hubs_scaled)

        # Here, we are storing distances for hubs within the same cluster and hubs belon to different clusters
        within_distances = []
        across_distances = []

        # Here, we are comparing unique pair of hubs
        for i in range(len(X_hubs_scaled)):
            for j in range(i+1, len(X_hubs_scaled)):
                distance = distance_matrix[i,j]
                if labels_k[i] == labels_k[j]:
                    within_distances.append(distance)
                else:
                    across_distances.append(distance)
        
        # Here, we are calculating the ratio of mean across clusters
        across_over_within = (sum(across_distances)/len(across_distances))/(sum(within_distances)/len(within_distances))

        # Here, we are counting hubs per clusters and sorting the cluster sizes in descending order
        cluster_sizes = pd.Series(labels_k).value_counts()
        cluster_sizes = sorted(cluster_sizes.to_list(), reverse = True)

        # Here, we are joining cluster sizes using hyphens
        cluster_sizes_text = "-".join(str(size) for size in cluster_sizes)

        # Here, we are storing the results for this K
        separation_results.append([k,across_over_within,cluster_sizes_text])
        print("K: ", k)
        print("Across over within ratio: ", across_over_within)
        print("Cluster sizes: ", cluster_sizes_text)

    separation_df = pd.DataFrame(separation_results, columns = ["K","across_over_within", "cluster_sizes"])

    # Here, we are saving the separation results
    separation_df.to_csv("outputs/part_d_separation.csv", index = False)
    print("Part D Separation Evidence: ")
    print(separation_df.to_string(index = False))

    # Here, we are attaching the K-Means cluster labels to the hub profile
    hub_profile_df["cluster"] = cluster_labels

    # Here, we are calculating the mean delay rate for each cluster
    cluster_delay_rates = (hub_profile_df.groupby("cluster")["delay_rate"].mean().sort_values())

    # Here, we are assigning segment names based on increasing delay rate
    segment_names = ["Standard","Watch","Escalate"]

    cluster_to_segment = dict(zip(cluster_delay_rates.index,segment_names))

    # Here, we are assigning the corresponding segment to each hub
    hub_profile_df["segment"] = (hub_profile_df["cluster"].map(cluster_to_segment))

    # Here, we are retrieving each hub's city and tier from the modelling data
    hub_details_df = (df[["hub_id","hub_city","hub_tier"]].drop_duplicates(subset = ["hub_id"]))

    # Here, we are combining the hub segment with its city and original tier
    hub_segments_df = hub_profile_df[["hub_id","segment"]].merge(hub_details_df, on = "hub_id", how = "left", validate = "one_to_one")

    # Here, we are saving the hub segments in ascending hub_id order
    hub_segments_df = hub_segments_df.sort_values("hub_id")
    hub_segments_df.to_csv("outputs/hub_segments.csv", index = False)

    print("Part D Hub Segments:")
    print(hub_segments_df.to_string(index = False))

    # Here, we are calculating segment statistics using hub-level averages
    segment_profile_df = (hub_profile_df.groupby("segment").agg(
        hubs = ("hub_id","size"), 
        mean_delivery_hours = ("mean_delivery_hours","mean"),
        delay_rate = ("delay_rate","mean"),
        mean_distance_km = ("mean_distance_km","mean"),
        mean_n_stops = ("mean_n_stops", "mean")
    ).reset_index())

    # Here, we are ordering the segments from lowest to highest delay rate
    segment_profile_df["segment"] = pd.Categorical(segment_profile_df["segment"],categories = segment_names, ordered = True)
    segment_profile_df = segment_profile_df.sort_values("segment").reset_index(drop = True)
    

    # Here, we are saving the segment-level operating statistics
    segment_profile_df.to_csv("outputs/segment_profile.csv", index = False)

    print("Part D Segment Profile: ")
    print(segment_profile_df.to_string(index = False))

    # Here, we are defining the expected segment for each hub tier
    tier_to_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}

    # Here, we are identifying hubs whose assigned segment differs from the segments implied by their original tier
    expected_segments = hub_segments_df["hub_tier"].map(tier_to_segment)

    mismatch_df = hub_segments_df.loc[hub_segments_df["segment"] != expected_segments].copy()

    # Here, we are saving only the mismatched hubs
    mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index = False)
    print("Part D Tier vs Segment Mismatches: ")
    print(mismatch_df.to_string(index = False))
    print("The total mismatched hubs: ", len(mismatch_df))

if __name__ == "__main__":
    main()