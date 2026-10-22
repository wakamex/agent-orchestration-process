# Documentation site

The site at [agentorchestrationprocess.com](https://agentorchestrationprocess.com/) is built with [Zola](https://www.getzola.org/) and the [zola-docs](https://github.com/wakamex/zola-docs) theme, vendored under `themes/zola-docs` at the commit in `THEME_COMMIT`. Its pages come only from the repository's `README.md` and `docs/*.md`: `ci/build_docs.py` writes them into `content/` with front matter and navigation, so edit the Markdown files, never `content/`.

```sh
./ci/install-zola.sh
python3 ci/build_docs.py
cd docs-site && ../.cache/zola/0.23.4/zola serve
```

`.github/workflows/docs-site.yml` builds the site on each push to `main` that changes the docs, fails on a broken internal link, and deploys it with `wrangler deploy` as a Cloudflare Worker serving static assets, configured in the repository's `wrangler.jsonc` for `agentorchestrationprocess.com` and `www.agentorchestrationprocess.com`.

To deploy by hand, put `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in the repository's `.env`, which Git ignores, and run `./deploy.sh`.
