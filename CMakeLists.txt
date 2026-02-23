# Tu nastavit verziu Qt6
set(Qt6_Version 6.8.2)

cmake_minimum_required(VERSION 3.16.0)

project(ImageSegmentation_APP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions( "/D_CRT_SECURE_NO_WARNINGS /MP /openmp" )
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(Qt6_DIR "D:/prog/Qt/${Qt6_Version}/msvc2022_64/lib/cmake/Qt6")
set(Qt6WidgetsTools_DIR "D:/prog/Qt/${Qt6_Version}/msvc2022_64/lib/cmake/Qt6WidgetsTools")
set(Qt6CoreTools_DIR "D:/prog/Qt/${Qt6_Version}/msvc2022_64/lib/cmake/Qt6CoreTools")
set(Qt6GuiTools_DIR "D:/prog/Qt/${Qt6_Version}/msvc2022_64/lib/cmake/Qt6GuiTools")

find_package(Qt6 COMPONENTS Widgets Core Gui REQUIRED)

# Specify OpenCV package location and find it
set(OpenCV_DIR "D:/prog/opencv4.1.2/opencv/build")
find_package(OpenCV REQUIRED)

# Add OpenCV include directories and link libraries to the target
# Use target-based commands so the include paths and libs are applied only to the executable


file(GLOB UI_FILES src/*.ui)
file(GLOB H_FILES src/*.h)
file(GLOB CPP_FILES src/*.cpp)
file(GLOB QRC_FILES src/*.qrc)

set(SOURCE_LIST ${CPP_FILES} ${UI_FILES} ${H_FILES} ${QRC_FILES})

add_executable(${PROJECT_NAME} ${SOURCE_LIST})

# Apply OpenCV include directories to the target
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link Qt and OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    ${OpenCV_LIBS}
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

# Deploy Qt using your existing windeployqt command
add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND "D:/prog/Qt/${Qt6_Version}/msvc2022_64/bin/windeployqt.exe" "$<TARGET_FILE:${PROJECT_NAME}>"
    COMMENT "Win deploy qt"
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
