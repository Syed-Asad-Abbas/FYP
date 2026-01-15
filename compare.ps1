
$csv1 = Import-Csv "file_list_1.csv"
$csv2 = Import-Csv "file_list_2.csv"

$dict1 = @{}
$csv1 | ForEach-Object { $dict1[$_.RelativePath] = $_ }

$dict2 = @{}
$csv2 | ForEach-Object { $dict2[$_.RelativePath] = $_ }

$onlyIn1 = @()
$onlyIn2 = @()
$modified = @()

# Check entries in 1
foreach ($path in $dict1.Keys) {
    if (-not $dict2.ContainsKey($path)) {
        $onlyIn1 += $path
    } else {
        # Check for modification (Size)
        if ($dict1[$path].Length -ne $dict2[$path].Length) {
            $modified += "$path (Size: $($dict1[$path].Length) vs $($dict2[$path].Length))"
        }
    }
}

# Check entries in 2
foreach ($path in $dict2.Keys) {
    if (-not $dict1.ContainsKey($path)) {
        $onlyIn2 += $path
    }
}

"=== Files Only in fyp_multimodal_model ===" | Out-File "comparison_report.txt"
$onlyIn1 | Sort-Object | Out-File "comparison_report.txt" -Append
"" | Out-File "comparison_report.txt" -Append

"=== Files Only in fyp_multimodal_model-laptop ===" | Out-File "comparison_report.txt" -Append
$onlyIn2 | Sort-Object | Out-File "comparison_report.txt" -Append
"" | Out-File "comparison_report.txt" -Append

"=== Modified Files (Different Size) ===" | Out-File "comparison_report.txt" -Append
$modified | Sort-Object | Out-File "comparison_report.txt" -Append
