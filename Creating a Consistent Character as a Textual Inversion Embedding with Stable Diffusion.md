# Creating a Consistent Character as a Textual Inversion Embedding with Stable Diffusion

copy from: https://web.archive.org/web/20230407215713/https://github.com/github.com/BelieveDiffusion/tutorials/blob/main/consistent_character_embedding/README.md#creating-a-consistent-character-as-a-textual-inversion-embedding-with-stable-diffusion

One of the great things about generating images with Stable Diffusion ("SD") is the sheer variety and flexibility of images it can output. However, some times it can be useful to get a consistent output, where multiple images contain the "same person" in a variety of permutations.

To that end, I've spent some time working on a technique for training Stable Diffusion to generate consistent made-up characters whose faces, bodies, and hair look essentially the same whenever you use them in a prompt. This tutorial is a description of the approach I use.

## LastName characters

You can see all of the "LastName" characters I've trained with this method on CivitAI. And credit where it's due - they were inspired by the Nobody series by Zovya. Thank you for the inspiration, Zovya!

![Antonia](https://github.com/carlosmizabi/stablediffusion/raw/main/imgs/lastname_antonia.jpg)


