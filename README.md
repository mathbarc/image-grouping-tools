# Image Grouping Tool

The current package contains tools for analysing big sets of images by leveraging CNN models as images descriptors. Currently the package contains a cli tool with the following commands:

```bash
image_grouping compute_features <path_to_image_folders>
```

The command above scans the appointed folder for PNG and JPEG images and generates a .pt file containing the features and image paths.

```bash
image_grouping apply_pca <path_to_features_file>
```

Applies the PCA algorithm for dimension reduction and stores it to a second file.

```bash
image_grouping cluster <path_to_features_file>
```

Clusterizes data and plots results.

For similarity search:

```bash
image_grouping compute_distances -i ../image-grouping-test/1/10.jpg -i ../image-grouping-test/1/09.jpg ./features.pt --dist cosine <default=euclidian>
```
This command will compute e distances, sort the images indexes according to distance and store the information on features.pt file

Getting the n more diverse images:

```bash
image_grouping  get_diversity features.pt -n 10
```

Get the 10 more diverse images on image set.

Finally, it is also possible to plot image diversity:

```bash
image_grouping plot_diversity features.pt
```
The diversity plot will be saved as diversity_plot.html


for a more detailed list of params it is possible to use the *--help* command with each of these options.
