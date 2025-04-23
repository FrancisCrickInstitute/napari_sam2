# Usage Guide
This will provide a few details on some features of the plugin and some things to consider.


## General Usage
### 1. Select the desired model and load it
The SAM2 models are, from smallest to largest: Tiny < Small < Base Plus < Large. Larger models will generally do better, but require more computation.

The "model folder" is where the models will be downloaded to (if not already present). Then, select a memory mode if using it (details [below](#low-memory-mode)), and load the model!

### 2. Select data
Any image layers in Napari will be available to select and calculate the embeddings for. Once this is complete, we can begin segmentation!

<!-- Custom reader stuff? -->

### 3. Adding prompts
You'll first need to creation the prompt layers via the "Create Layers" button. The "Point Prompts" layer will be the main layer where you add prompts from.

Here, a left-click is a positive prompt, and a right-click is a negative prompt. The label ID is the label currently selected in the corresponding "Masks" `Labels` layer. Note that you can adjust this from the "Point Prompts" layer by using the `+` and `-` keys to increment/decrement the label respectively. Using the `M` key will also go to the next available/unused label ID.

By default, the segmentations will be created instantaneously on click. By disabling "Auto-Segment on Click", you can manually trigger segmentation based on the provided prompts with the "Trigger Segmentation" button.

#### 3.1 Storing/Loading Prompts
Using the "Store Layers" button will save the current "Point Prompts" layer as a CSV file.

The "Load Layers" button can then be used to load the CSV of point prompts, which will then trigger a segmentation using these points. Note that results can differ here as discussed [here](#click-by-click-auto-segment-vs-batching-prompts-manual-trigger) depending on prompt order.

### 4. Propagation
If you wish to track objects, or just propagate the segmentation across a Z- or time-axis, you can use SAM2's propagation. From the earliest frame/slice where a prompt exists, masks will be propagated until the end of the video/image stack. You can change this direction with the "Reverse Propagation" checkbox. You can also cancel anytime (such as if you notice that more prompting is needed) with the "Cancel Propagation" button.

This facilitates a human-in-the-loop interaction, where you can prompt, propagate, re-prompt, re-propagate etc. to speed up annotation, tracking, or whatever other use case you have.

### 5. Export Masks
The resulting masks can be exported to TIFF using the "Export Masks" button. Alterantively, for visualization you can export an overlay, which will create a mp4 video of the masks overlaid on top of the source image, using the provided opacity and FPS.

While Napari currently provides limited support for the `Tracks` layer, this can also be created to show the movement of objects over time. For each object, the centroid is used for its location at each timepoint/slice.

## Further Details
### Click-by-click (Auto-Segment) vs Batching Prompts (Manual Trigger)
You may occasionally notice differencess in the segmentation, most likely when the same prompts are used in the auto-segment and manual segment modes. These are not equivalent, as in auto-segment mode all clicks after the first are effectively conditioned on the previous clicks (and the resulting masks), rather than considered simultaneously as they are in the manual mode (as all prompts are batched together).

This can also lead to discrepancies in other scenarios where prompts are considered in different contexts:
- When loading previously saved prompts, these will be batched for each object rather than considered one-by-one, which may lead to different results if they were initially added in auto-segment mode
- When deleting point prompts, in the re-segment all previous prompts are now considered jointly, rather than sequentially as they would have been in auto-segment mode. Therefore, deleting a prompt may not be an "undo" and revert back to the *exact* same mask as before the addition of the deleted prompt.

### Low-Memory Mode
As noted in *many* GitHub issues (e.g. this one), the memory usage of SAM2 scales with the number of frames in a video (or slices for image stacks). To counter this, there is an option for a "low-memory mode" and a "super low-memory mode". Note that this is primarily to improve performance for propagation (rather than just segmenting images), but is useful in both cases.

The "low-memory mode" does the following things:
1. Clears memory in a window around completed frames to avoid accumulation over time (only applies to propagation)
2. Enables the `offload_video_to_cpu` argument to avoid storing frames on the GPU
3. Enables the `async_loading_frames` argument in combination with a source-code modification to avoid storage [details](https://github.com/facebookresearch/sam2/issues/264)

In addition to the above, the "super low-memory mode" offloads the inference state to the CPU (via the `offload_state_to_cpu` argument). Note that this will notably decrease performance, so should be a last resort.