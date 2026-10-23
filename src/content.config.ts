import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string().optional(),
    description: z.string().optional(),
    pubDate: z.coerce.date(),
    dateRange: z.string().optional(),
    tags: z.array(z.string()).default([]),
  }),
});

export const collections = { blog };
