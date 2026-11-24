variable "region" {
  type    = string
  default = "us-east-1"
}

variable "lambda_zip" {
  type    = string
  default = "../build/lambda.zip"
}

variable "model_version" {
  type    = string
  default = "lstm-v1"
}

variable "budget_email" {
  description = "Email for the $1 budget alert (leave empty to skip)."
  type        = string
  default     = ""
}
