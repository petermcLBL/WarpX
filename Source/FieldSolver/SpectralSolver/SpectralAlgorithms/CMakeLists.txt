target_sources(WarpX
  PRIVATE
    GalileanAlgorithm.cpp
    PMLPsatdAlgorithm.cpp
    PsatdAlgorithm.cpp
    SpectralBaseAlgorithm.cpp
    AvgGalileanAlgorithm.cpp
)

if(WarpX_DIMS STREQUAL RZ)
    target_sources(WarpX
      PRIVATE
        SpectralBaseAlgorithmRZ.cpp
        PsatdAlgorithmRZ.cpp
	)

endif()

if ( WarpX_USE_FULL_SPIRAL STREQUAL ON )
    target_sources ( WarpX
	PRIVATE 
	psatd.fftx.source.cpp
	)
    target_include_directories ( WarpX PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )
##    set_source_files_properties ( psatd.fftx.source.cpp PROPERTIES COMPILE_OPTIONS "-I${CMAKE_CURRENT_SOURCE_DIR}" )
endif ()
