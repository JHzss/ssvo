file(REMOVE_RECURSE
  "bin/monoVO_euroc.pdb"
  "bin/monoVO_euroc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/monoVO_euroc.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
