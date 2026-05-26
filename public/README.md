# public/

Static assets served by Chainlit.

## Contents

| File / Folder | Description |
|---|---|
| `stylesheet.css` | Custom CSS overrides for the Chainlit UI |
| `logo_dark.svg` | Logo shown on dark background |
| `logo_light.svg` | Logo shown on light background |
| `icons/` | Favicon and app icons |

These files are mounted into the Docker container at `/app/public/` via the volume binding in `docker-compose.yml`.
