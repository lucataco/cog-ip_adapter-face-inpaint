# tencent-ailab / IP-Adapter

This is an implementation of a face-inpaint method [tencent-ailab / IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i face_image=@ai_face2.png -i source_image=@geisha.png

## Example:

![alt text](output.0.png)
