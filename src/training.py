import numpy as np
import torch

# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(
    model, chunksize,
    height, width, focal_length, tform_cam2world,
    near_thresh, far_thresh, depth_samples_per_ray,
    encoding_function, get_minibatches_function
):
    # Get the "bundle" of rays through all image pixels.
    ray_origins, ray_directions = get_ray_bundle(height, width, focal_length,
                                                tform_cam2world)

    # Sample query points along each ray
    query_points, depth_values = compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
    )

    # "Flatten" the query points.
    flattened_query_points = query_points.reshape((-1, 3))

    # Encode the query points (default: positional encoding).
    encoded_points = encoding_function(flattened_query_points)

    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    batches = get_minibatches_function(encoded_points, chunksize=chunksize)
    predictions = []
    for batch in batches:
        predictions.append(model(batch))
    radiance_field_flattened = torch.cat(predictions, dim=0)

    # "Unflatten" to obtain the radiance field.
    unflattened_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flattened, unflattened_shape)

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, _, _ = render_volume_density(radiance_field, ray_origins, depth_values)

    return rgb_predicted

def train_tinynerf(
    model,
    images,
    tform_cam2world,
    target_tform_cam2world,
    height, width, focal_length, near_thresh, far_thresh,
    device,
    num_encoding_functions=6,
    encode=lambda x: positional_encoding(x, num_encoding_functions=6),
    depth_samples_per_ray = 32,
    chunksize = 16384,
    lr = 5e-3,
    num_iters = 1000
):
    """
    Train-Eval-Repeat!
    """


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Lists to log metrics etc.
    psnrs = []
    iternums = []

    for i in range(num_iters):
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        rgb_predicted = run_one_iter_of_tinynerf(model, chunksize, height, width, focal_length,
                                                target_tform_cam2world, near_thresh,
                                                far_thresh, depth_samples_per_ray,
                                                encode, get_minibatches)

        # Compute mean-squared error between the predicted and target images. Backprop!
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()