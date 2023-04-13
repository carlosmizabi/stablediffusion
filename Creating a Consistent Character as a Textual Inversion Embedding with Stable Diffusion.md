# Creating a Consistent Character as a Textual Inversion Embedding with Stable Diffusion

copy from: https://web.archive.org/web/20230407215713/https://github.com/github.com/BelieveDiffusion/tutorials/blob/main/consistent_character_embedding/README.md#creating-a-consistent-character-as-a-textual-inversion-embedding-with-stable-diffusion

One of the great things about generating images with Stable Diffusion ("SD") is the sheer variety and flexibility of images it can output. However, some times it can be useful to get a consistent output, where multiple images contain the "same person" in a variety of permutations.

To that end, I've spent some time working on a technique for training Stable Diffusion to generate consistent made-up characters whose faces, bodies, and hair look essentially the same whenever you use them in a prompt. This tutorial is a description of the approach I use.

## LastName characters

You can see all of the "LastName" characters I've trained with this method on CivitAI. And credit where it's due - they were inspired by the Nobody series by Zovya. Thank you for the inspiration, Zovya!

<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/lastname_antonia.jpg" alt="Antonia LastName" style="max-width: 100%;" width="256" height="384">
<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs//lastname_brie.jdg" alt="Brie LastName" style="max-width: 100%;" width="256" height="384">
<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/lastname_caoimhe.jpg" alt="Caoimhe LastName" style="max-width: 100%;" width="256" height="384">
<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/lastname_denise.jpg" alt="Denise LastName" style="max-width: 100%;" width="256" height="384">
<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/lastname_elise.jpg" alt="Elise LastName" style="max-width: 100%;" width="256" height="384">

## Goals

If all goes to plan, by the end of this tutorial you will have created a Stable Diffusion
[Textual Inversion embedding](https://arxiv.org/abs/2208.01618) that can reliably recreate a consistent character 
across multiple poses, SD checkpoints, hair styles, body types, and prompts.

## Process

The creation process is split into five steps:

- Generating input images
- Filtering input images
- Tagging input images
- Training an embedding on the input images
- Choosing and validating a particular iteration of the trained embedding

## 1. Generating input images

Training an AI is a classic example of ["garbage in, garbage out"](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out).
The better the input images you provide, 
the better the output you'll get. To that end, I use SD to generate the input images for the character I'm going 
to train. That way, I can generate hundreds of permutations based on a description of that character, and pick just 
the best images to use for the later training.

I generate these images via [Automatic1111's Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
(I'll call it "A1111" below). I won't cover 
how to set up A1111 here; there are lots of tutorials available for getting A1111 up and running. But here's how 
I customize A1111 for input image generation.

### Choosing a checkpoint for generating your input images

You can use any SD checkpoint you like to generate your input images, although it's essential
that the model you choose has seen (and can generate) representative examples like your character. 
I've been creating photorealistic made-up characters, and I've found 
[Deliberate](https://civitai.com/models/4823/deliberate) (v2) to be a good, flexible checkpoint for that, 
but there are plenty of other models available on sites like 
[CivitAI](https://web.archive.org/web/20230407215713/https://civitai.com/).

### Turning on the inclusion of tags in the output filenames

We're going to define an input prompt for generating our input images, which we'll convert into a training 
prompt later on. To make that conversion process as easy as possible later, set the "Images filename pattern" i
n A1111's "Settings > Saving images/grids" settings to `[seed]-[prompt_spaces]`, 
and then click "Apply settings". This will include the seed and generation prompt in the filename 
of every generated PNG image we create below.

### Setting up an input prompt

With your preferred generation checkpoint selected in A1111's web interface, open the `txt2img` tab, and enter a 
positive prompt with the following format:

```
an extreme closeup front shot photo of %your character's look% (naked:1.3), %your character's body shape%, %your character's hairstyle%, (neutral gray background:1.3), neutral face expression
```

For this input prompt, I've boosted the strength of the `naked` prompt token, to say "we really want the images 
to be of the character naked, please." I've found this reduces the number of non-naked images we have to throw 
away later on. Similarly, I've boosted "neutral gray background", so that (hopefully) all of the images come 
out with nothing distracting in the background.

Replace `%your character's look%`, `%your character's body shape%`, and `%your character's hairstyle%`
in the prompt with descriptions that will reliably generate a face, body, and hairstyle that match your target character.

Be descriptive, but try to stay within the `75/75` limit of A1111's input prompt box. The fewer essential details y
ou provide, the more chance there is that those details will be present in every image you generate. 
(And the facial details are the ones that really matter.)

One tip I've learned from experience: adding a country of origin can really help to hone in on an overall base look,
which you can then refine with more details. (A country of origin may also guide your character's overall skin tone, 
too.)

Here's an example of just-enough-but-not-too-much detail:

```
an extreme closeup front shot photo of a beautiful 25yo French woman with defined cheekbones, a straight nose, full lips, hazel eyes, chin dimple, square jaw, plucked eyebrows, (naked:1.3), small breasts, toned body, chin length straight black hair in a bob cut with bangs, (neutral gray background:1.3), neutral face expression
```

For the negative prompt, try something like this:

```
(gray hair:1.3), (glasses:1.2), (earrings:1.2), (necklace:1.2), (high heels:1.2), young, loli, teen, child, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, tattoo
```

Next, set the following `txt2img` settings in A1111:

- Sampling method: DPM++ 2M Karras
- Sampling steps: 30 
- Restore Faces: On 
- Tiling: Off 
- Hires. fix: Off 
- Width: 512 
- Height: 512 
- CFG Scale: 7 
- Seed: -1

These settings should give good-quality outputs, at the expense of slightly longer generation times. (But remember: garbage in, garbage out.)

### Why naked and neutral?

I like to make input images that are as "neutral" as possible, so that SD learns the essence of them without also 
learning things that we might want to change for variety in image generation prompts. So, I try to avoid generating 
images that contain things like glasses, earrings, necklaces, and so on, that might bias later generation to 
include those same items.

I also choose to generate the training input images without clothes, because I want to train the base concept 
of a hypothetical human that I can then add any clothing or accessories to via custom prompts later on. SD has seen 
a lot of humans wearing a lot of different clothes, but it has never seen your custom character naked, so that's what 
we'll give it as input, for the most flexibility.

I also use a neutral gray background in all my input images, to keep the training focused on the character on the
foreground. (We'll tell SD that we used a neutral gray background later on, so that it doesn't learn "neutral gray 
background" as part of the character's attributes.)


### Testing the input prompt

With the settings above in place, generate a small set of test images to see how well your prompt performs. 
I usually set `Batch count` to 2 and `Batch size` to 4 (with a `Seed` of `-1`), then hit Generate, to create eight test 
images. This helps me to see if the prompt creates a consistent (enough) output across multiple seeds.

Note that even with a detailed description like the one above, your character might not (yet) look entirely 
consistent between all of the images. That's okay - we will improve that by filtering the images later, and also by 
averaging the character's visual characteristics through training. But you still want to be seeing a recognizable-enough 
consistency at this stage. If you don't, tweak your prompt, and try again.

### Generating permutations

The next thing we want to do is to generate a whole bunch of variations of our character, with different 
viewing angles and camera zooms. I mentioned above that SD follows the principle of "garbage in, garbage out"; 
the same also holds for "variation in, variation out". In other words, the more of a variety of angles and framings
we can show SD of our character, the better SD will become at generating varied angles and framings when we use our
character in prompts later on.

To add this variety, open the `Script` menu, and select `X/Y/Z Plot`. Set `X type` and 
`Y type` to `Prompt S/R` ("Prompt search and replace"), and keep `Z type` as Nothing.

In the `X values` text box, paste the following five zoom levels:

```
an extreme closeup, a medium closeup, a closeup, a medium shot, a full body
```

This tells the A1111 web UI that we want to generate images with five different permutations 
of our prompt - one that uses the original `an extreme closeup` text from the main prompt box, and 
four others that replace that text in the prompt with an alternative zoom level. This will give us 
images of our character from a variety of distances.

In the `Y values` text box, paste the following five viewing angles:

```
front shot, rear angle, side angle, shot from above, low angle shot
```

This tells the A1111 web UI that we want to generate images with five more permutations of our 
prompt - one that uses the original `front shot` viewing angle, and four others that replace that 
text with an alternative angle for viewing the character.

Because we have two `Prompt S/R` options set for the script, with five variations in each, we've 
actually told A1111 to generate 25 permutations of our prompt - one for each combination of 
framing and viewing angle. This will give us lots of varieties of views of our character to 
choose from for training.

(Note: I borrowed these zoom levels and viewing angles from the [Unstable Diffusion tagging white paper](https://web.archive.org/web/20230407215713/https://docs.google.com/document/d/1-DDIHVbsYfynTp_rsKLu4b2tSQgxtO5F6pNsNla12k0/edit).)

Finally, check the box next to `Keep -1 for seeds`, and set `Batch Count` and `Batch Size` both to 4. 
This will generate 16 images for every permutation of the above, in batches of four for speed. 
(If your GPU can handle it, you can use a `Batch Count` of 2 and a `Batch Size` of 8 instead.)

Here's how that looks for me with today's A1111 interface:

<img src="https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/step_1_input_generation_settings.jpg" alt="Input generation settings" style="max-width: 100%;">