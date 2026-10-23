# LinkedIn Post: The Silent Bug & Digit Duel

*Copy and paste the text below directly into LinkedIn. It's written in an authentic developer voice without AI clichés, buzzwords, or synthetic cheerleading.*

---

I spent two hours yesterday scratching my head over why a 2-layer RNN refused to learn MNIST digits.

MNIST is typically treated as a solved benchmark. Throw a basic 3-layer CNN at it and you hit 99% accuracy in under five minutes. But I wanted to try something different: what if we treat an image strictly as sequential data?

Instead of looking at the 2D grid all at once, the RNN reads the image row-by-row — 28 sequential steps, with 28 pixels at each step.

I spun up the training loop. 
Epoch 1 loss: 1.77. 
Epoch 10 loss: 1.53. 
Test accuracy: stuck at 84.7%.

It wasn't overfitting. It was barely learning.

I went line-by-line through the tensor shapes, learning rates, and hidden dimensions. Everything looked normal until I checked line 296:

`output = torch.sigmoid(model(images))`  
`loss = criterion(output, labels)`  

There it was. 

PyTorch’s `nn.CrossEntropyLoss` already combines `LogSoftmax` and negative log-likelihood under the hood. It expects raw, unnormalized logits. By wrapping the model output in a `torch.sigmoid()`, I was compressing the logits into a (0, 1) range before feeding them to the loss function. The gradients were getting crushed before they even had a chance to backpropagate. The network was essentially suffocating.

I deleted that single `torch.sigmoid()` call and ran it again.

The loss collapsed from 0.60 down to 0.10. 
Test accuracy instantly jumped from 84.7% to 97.0%. No architecture changes, no hyperparameter tuning — just removing one wrong function call.

That debugging session turned into a complete project: **Digit Duel**. 

I pitted the spatial CNN (99.25% accuracy) against the sequential RNN (97.00% accuracy) to explore how differently they interpret handwritten shapes.

Then came the deployment dilemma. 

I wanted anyone to be able to test both models on their own handwriting, but free container platforms like Streamlit Community Cloud and Render have one frustrating flaw: they sleep after inactivity. If someone clicks a link from a post, having them stare at a "Waking up server (estimated time: 50s)" screen kills the experience.

So I took a different approach:
1. Exported the PyTorch checkpoints to a tiny 1.5 MB standalone ONNX file.
2. Built a lightweight frontend using HTML5 Canvas and ONNX Runtime Web (WebAssembly).
3. Deployed it as a static app on Vercel.

Result: 100% free hosting, zero cold starts, and predictions execute in about 3 milliseconds directly inside the user's browser. 

There was one more catch: user-drawn digits usually perform terribly on MNIST models (~40% accuracy) if you just naively downscale the canvas. Real MNIST digits are cropped to their bounding box, scaled to fit a 20×20 frame, and translated so their center-of-mass sits at (13.5, 13.5). Implementing that exact centering logic inside the JavaScript canvas brought real-world drawing accuracy right in line with the test set.

You can draw a digit on the canvas, upload a photo, or turn on "Duel Mode" to watch both networks compete in real time.

Live demo: [Insert your Vercel link here once deployed]  
GitHub repo: https://github.com/fireside4137/Digit_Duel  

Curious for those who've trained models from scratch — what’s the subtlest silent bug you’ve ever had to hunt down in a training loop?

#MachineLearning #DeepLearning #PyTorch #WebML #ONNX #ComputerVision
