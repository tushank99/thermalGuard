terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.0" }
  }
  # Local state; for a team setup use an S3 backend.
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "thermalguard", managed_by = "terraform" } }
}

data "aws_caller_identity" "me" {}

# ---- Artifact bucket (Lambda zip > 50 MB must go through S3) ----
resource "aws_s3_bucket" "artifacts" {
  bucket        = "thermalguard-artifacts-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "lambda_zip" {
  bucket = aws_s3_bucket.artifacts.id
  key    = "lambda/lambda-${filemd5(var.lambda_zip)}.zip"
  source = var.lambda_zip
  etag   = filemd5(var.lambda_zip)
}

# ---- IAM: least privilege, only CloudWatch logs ----
resource "aws_iam_role" "lambda" {
  name = "thermalguard-api-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "lambda.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/thermalguard-api"
  retention_in_days = 14 # GDPR note: short retention for request logs
}

# ---- Lambda + Function URL (free; no API Gateway needed) ----
resource "aws_lambda_function" "api" {
  function_name    = "thermalguard-api"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.12"
  handler          = "thermalguard.serving.app.handler"
  s3_bucket        = aws_s3_bucket.artifacts.id
  s3_key           = aws_s3_object.lambda_zip.key
  source_code_hash = filebase64sha256(var.lambda_zip)
  memory_size      = 1024
  timeout          = 15
  architectures    = ["x86_64"]
  environment {
    variables = {
      MODEL_DIR     = "/var/task/model"
      MODEL_VERSION = var.model_version
    }
  }
  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy_attachment.logs]
}

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  cors {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
    allow_headers = ["content-type"]
  }
}

# Public Function URLs also need lambda:InvokeFunction, restricted to calls that come through the URL
# (direct Invoke API calls stay denied).
resource "aws_lambda_permission" "url_invoke" {
  statement_id             = "FunctionURLInvokeAllowPublicAccess"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.api.function_name
  principal                = "*"
  invoked_via_function_url = true
}

# ---- Cost guard-rail: email alert if monthly spend forecast exceeds $1 ----
resource "aws_budgets_budget" "guard" {
  count        = var.budget_email == "" ? 0 : 1
  name         = "thermalguard-1usd"
  budget_type  = "COST"
  limit_amount = "1"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
