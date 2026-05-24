import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

// TODO: change `site` to your GitHub Pages URL (e.g. https://<user>.github.io)
// TODO: change `base` to your repo name. If you publish from a user/organization
//       page (e.g. <user>.github.io), set base to '/' instead.
export default defineConfig({
  site: 'https://danielaraya.github.io',
  base: '/porfolio',
  integrations: [react(), tailwind()],
});
