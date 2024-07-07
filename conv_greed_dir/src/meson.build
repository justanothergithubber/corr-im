common_sources = [
    'cvar.cpp',
    'graph.cpp',
    'inf_max.cpp',
]

conv_greed_sources = common_sources + ['conv_greed.cpp']

pmc_greed_sources = common_sources + ['pmc_greed.cpp']

cvar_greed_sources = common_sources + ['cvar_greed.cpp']

cvar_est_sources = common_sources + ['cvar_est.cpp']

pmc_est_sources = common_sources + ['pmc_est.cpp']

common_args = {
    'cpp_args': ['-march=native'],
    'link_args': ['-static', '-static-libgcc', '-static-libstdc++'],
    'include_directories': inc,
    'native': true
}


executable(
    'conv_greed',
    conv_greed_sources,
    kwargs: common_args
)

executable(
    'pmc_greed',
    pmc_greed_sources,
    kwargs: common_args
)

executable(
    'cvar_greed',
    cvar_greed_sources,
    kwargs: common_args
)

executable(
    'cvar_est',
    cvar_est_sources,
    kwargs: common_args
)

executable(
    'pmc_est',
    pmc_est_sources,
    kwargs: common_args
)