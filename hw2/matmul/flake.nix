{
  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = nixpkgs.legacyPackages.${system}; in
      {
        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.default ];
          nativeBuildInputs = with pkgs; [ valgrind ];
        };
        packages.default = pkgs.stdenv.mkDerivation (finalAttrs: {
          name = "cfds-hw2";
          src = ./.;

          nativeBuildInputs = with pkgs; [ ];
          buildInputs = with pkgs; [ mpi ];

          installPhase = ''
            runHook preInstall
            install -Dm755 main $out/main
            runHook postInstall
          '';
        });
      }
    );
}
