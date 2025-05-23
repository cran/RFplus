test_that("RFplus works with different methods and included data", {

  Covariates = list(
    MSWEP = terra::rast(system.file("extdata/MSWEP.nc", package = "RFplus")),
    CHIRPS = terra::rast(system.file("extdata/CHIRPS.nc", package = "RFplus")),
    DEM = terra::rast(system.file("extdata/DEM.nc", package = "RFplus"))
  )

  BD_Insitu = data.table::fread(system.file("extdata/BD_Insitu.csv", package = "RFplus"))
  Cords_Insitu = data.table::fread(system.file("extdata/Cords_Insitu.csv", package = "RFplus"))

  # Test with "RQUANT" method
  result_quant = RFplus(BD_Insitu, Cords_Insitu, Covariates, n_round = 1, wet.day = 0.1,
                        ntree = 2000, seed = 123, training = 0.8, Rain_threshold = list(no_rain = c(0, 1),
                                                                                        light_rain = c(1, 5),
                                                                                        moderate_rain = c(5, 20),
                                                                                        heavy_rain = c(20, 40),
                                                                                        violent_rain = c(40, 100)),
                        method = "RQUANT", ratio = 10, save_model = FALSE, name_save = NULL)
  expect_true(inherits(result_quant, "list"))
  expect_true(inherits(result_quant$Ensamble, "SpatRaster"))
})
