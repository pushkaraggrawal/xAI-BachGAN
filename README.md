## xAI-BachGAN

# xAI
Using DeepSHAP to mask gradients generated from the output of the PatchGAN generator. Used to deliver a more information rich feedback through gradients.
Achieved by altering the backward_hook function in torch.nn class.

# BachGAN
Model that generates images using inputs of class labeled saliency maps rather than the traditional segmented inputs. A background retrieval module hallucinates a background for the foreground objects. The hallicinated background is passed through to the traditional GAN architectures. Available modes use Pix2pix, Pix2pixHD, SPADE and ResNet architectures.

# Citations
Original works on BachGAN: https://github.com/Cold-Winter/BachGAN
Original works on xAIGAN: https://github.com/explainable-gan/XAIGAN
