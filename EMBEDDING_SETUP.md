# Embedding Files Setup Guide

The embedding files (300MB total) are too large for GitHub's free tier. You need to host them separately and update the download URLs in the app.

## Files to Host:
- `embeddings_8` (12MB)
- `embeddings_bi_grams` (133MB) 
- `embeddings_8.wv.vectors.npy` (72MB)
- `embeddings_8.trainables.syn1neg.npy` (72MB)

## Free Hosting Options:

### Option 1: Google Drive (Recommended)
1. Upload files to Google Drive
2. Set sharing to "Anyone with the link can view"
3. Get direct download links using format:
   `https://drive.google.com/uc?export=download&id=FILE_ID`
4. Update URLs in `streamlit_app.py` lines 106-110

### Option 2: Dropbox
1. Upload to Dropbox
2. Get public links and modify to direct download:
   - Change `?dl=0` to `?dl=1` at the end
3. Update URLs in `streamlit_app.py`

### Option 3: OneDrive
1. Upload to OneDrive
2. Generate sharing links
3. Modify links for direct download access
4. Update URLs in `streamlit_app.py`

### Option 4: GitHub Release Assets
1. Create a release in your GitHub repo
2. Upload files as release assets (up to 2GB per file)
3. Use release asset URLs
4. Update URLs in `streamlit_app.py`

## How to Update URLs:

Edit `streamlit_app.py` lines 105-110:

```python
embedding_urls = {
    'embeddings_8': 'YOUR_DIRECT_DOWNLOAD_URL_HERE',
    'embeddings_bi_grams': 'YOUR_DIRECT_DOWNLOAD_URL_HERE', 
    'embeddings_8.trainables.syn1neg.npy': 'YOUR_DIRECT_DOWNLOAD_URL_HERE',
    'embeddings_8.wv.vectors.npy': 'YOUR_DIRECT_DOWNLOAD_URL_HERE'
}
```

## Testing:
1. Deploy app to Streamlit Community Cloud without embedding files
2. First-time users will see download progress bars
3. Files are cached after first download

## Alternative: Smaller Models
Consider training smaller Word2Vec models to reduce file sizes if download speed becomes an issue.