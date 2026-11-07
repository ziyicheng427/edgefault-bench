# Zenodo Archival Steps (DOI for v1.2.0)

Status: ** awaiting maintainer action** (requires a Zenodo account; the
repository side is ready).

1. Sign in at <https://zenodo.org> (an ORCID login works).
2. Open **Account Settings → GitHub** and authorize Zenodo's GitHub
   integration.
3. Enable the **ziyicheng427/edgefault-bench** repository in the integration
   list.
4. Zenodo imports **GitHub releases**. The
   [v1.2.0 release](https://github.com/ziyicheng427/edgefault-bench/releases/tag/v1.2.0)
   already exists; if it is not picked up automatically, create an empty
   follow-up release (or re-publish v1.2.0) to trigger the harvest.
5. In the Zenodo record: verify metadata (creator Ziyi Cheng; license
   Apache-2.0 for code — note Zenodo archives the repository snapshot, and
   dataset files are NOT redistributed), then **Publish** to mint the DOI.
6. Afterwards in this repository:
   - add the DOI to `CITATION.cff` (`preferred-citation` or
     `identifiers`),
   - replace the archived-release checklist entry in `ROADMAP.md` (Milestone 3
     "Archive an eligible release with a persistent identifier"),
   - add the DOI to `paper/main.tex` (availability statement) and recompile.

Zenodo keeps each release as a new version of one concept DOI; cite the
concept DOI for "the software" and the version DOI for v1.2.0 specifically.
