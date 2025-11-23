# GitHub Setup Guide

Your repository is ready to push to GitHub! Follow these steps:

## Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Choose a repository name (e.g., "regression-models" or "energy-consumption-prediction")
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Add Remote and Push

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

### Option A: If you haven't created the repo yet, use these commands:

```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option B: If you already created the repo, GitHub will show you the exact commands

Copy and paste the commands from GitHub's setup page.

## Step 3: Update Git Configuration (Optional but Recommended)

To set your global git identity (so you don't have to set it for each repo):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Troubleshooting

### If you get authentication errors:
- GitHub now requires a Personal Access Token instead of password
- Go to: Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate a new token with `repo` permissions
- Use the token as your password when pushing

### If you want to change the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### To check your remote:
```bash
git remote -v
```

