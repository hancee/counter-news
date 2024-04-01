#!/bin/bash

set -eux

# Clone git repo
echo "Cloning git repo..."
export GIT_REPO_URL="github.com/hancee/counter-news.git"
export GIT_BRANCH="main"
export GIT_USERNAME="stat299"
export GIT_ACCESS_TOKEN=$(aws secretsmanager get-secret-value --secret-id git-access-token --query SecretString --output text | jq -r '.password')
git -C /home/ec2-user/SageMaker clone --branch ${GIT_BRANCH} https://${GIT_USERNAME}:${GIT_ACCESS_TOKEN}@${GIT_REPO_URL}
echo "Successfully cloned $GIT_REPO_URL/$GIT_BRANCH"

#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF' # Initialize new shell instance with sudo rights
unset SUDO_UID # Disable sudo rights for next lines

# Install a separate conda installation via Miniconda
WORKING_DIR=/home/ec2-user/SageMaker/custom-miniconda
mkdir -p "$WORKING_DIR"
wget https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda" 
rm -rf "$WORKING_DIR/miniconda.sh"
echo "Successfully installed miniconda."

# Create a custom conda environment
source "$WORKING_DIR/miniconda/bin/activate"
KERNEL_NAME="counter-news"
PYTHON="3.9"
conda create --yes --name "$KERNEL_NAME" python="$PYTHON"
conda activate "$KERNEL_NAME"
pip install --quiet ipykernel
echo "Successfully created custom conda environment."

# Set up poetry-managed environment
echo "Setting up poetry-managed environment..."
curl -sSL https://install.python-poetry.org | python3 -
cd SageMaker/$KERNEL_NAME
# poetry env use "$WORKING_DIR/miniconda/envs/$KERNEL_NAME"  # Note use system if using newly created conda environment
poetry env use system
poetry config installer.max-workers 10
poetry install
poetry --version
echo "Successfully set up poetry-managed environment."

# Register the environment as a Jupyter kernel
# poetry run python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$KERNEL_NAME"
# echo "Registered $KERNEL_NAME as kernel."

EOF
