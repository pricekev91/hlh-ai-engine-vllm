variable "pm_api_url" {
  description = "Proxmox API URL"
  type        = string
}

variable "pm_api_token_id" {
  description = "Proxmox API token ID"
  type        = string
}

variable "pm_api_token_secret" {
  description = "Proxmox API token secret"
  type        = string
  sensitive   = true
}

variable "target_node" {
  description = "Proxmox node name"
  type        = string
  default     = "prox01"
}

variable "hostname" {
  description = "LXC hostname"
  type        = string
  default     = "hlh-ai-engine-vllm"
}

variable "vmid" {
  description = "LXC VMID"
  type        = number
  default     = 113
}

variable "ostemplate" {
  description = "Proxmox LXC template"
  type        = string
  default     = "local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst"
}

variable "storage" {
  description = "Proxmox storage for rootfs"
  type        = string
  default     = "RaidZ1-6TB"
}

variable "rootfs_size_gb" {
  description = "Root filesystem size in GB"
  type        = number
  default     = 64
}

variable "cores" {
  description = "CPU cores"
  type        = number
  default     = 12
}

variable "memory" {
  description = "Memory in MB"
  type        = number
  default     = 49152
}

variable "swap" {
  description = "Swap in MB"
  type        = number
  default     = 4096
}

variable "bridge" {
  description = "Network bridge"
  type        = string
  default     = "vmbr0"
}

variable "ip_cidr" {
  description = "Static IP in CIDR format"
  type        = string
  default     = "192.168.1.13/24"
}

variable "gateway" {
  description = "Gateway"
  type        = string
  default     = "192.168.1.1"
}

variable "network_tag" {
  description = "VLAN tag (0 means untagged)"
  type        = number
  default     = 0
}

variable "lxc_root_password" {
  description = "Optional root password for LXC"
  type        = string
  default     = ""
  sensitive   = true
}

variable "description" {
  description = "LXC description (ROCm version is unpinned, never pinned — set via deploy-hlh-ai-engine-vllm.sh ROCM_VERSION env, default 10.0.0 2026-08-26)"
  type        = string
  default     = "vLLM AI engine ROCm gfx1150 (vLLM + Open WebUI, never pinned) model storage on RaidZ1-6TB"
}

variable "model_mount_path" {
  description = "Host path for model storage mount point inside LXC"
  type        = string
  default     = "/srv/ai/models"
}

variable "model_storage" {
  description = "Proxmox storage pool for model volume"
  type        = string
  default     = "RaidZ1-6TB"
}
