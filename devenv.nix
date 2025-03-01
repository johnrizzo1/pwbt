{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs; [ 
    git
    # ta-lib
    # (python3.withPackages(ps: with ps; [ 
    #   python-dotenv
    #   pandas
    #   numpy
    #   sqlalchemy
    #   psycopg
    # ]))
  ];
  languages.python.enable = true;
  languages.python.venv.enable = true;
  languages.python.uv.enable = true;
  languages.python.uv.sync.enable = true;

  enterShell = ''
    git --version
  '';

  dotenv.enable = true;
  difftastic.enable = true;
  # See full reference at https://devenv.sh/reference/options/
}
