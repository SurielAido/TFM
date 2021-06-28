class Utils:

    @staticmethod
    def class_2_string(numeric_class):
        subclass = ''
        if numeric_class == 0:
            subclass = "Abduction hostage"
        elif numeric_class == 1:
            subclass = "Action"
        elif numeric_class == 2:
            subclass = "Adventure"
        elif numeric_class == 3:
            subclass = "Animal"
        elif numeric_class == 4:
            subclass = "Animation"
        elif numeric_class == 5:
            subclass = "Beach, sea"
        elif numeric_class == 6:
            subclass = "Bomb explosion"
        elif numeric_class == 7:
            subclass = "Car chase"
        elif numeric_class == 8:
            subclass = "Child (ren)"
        elif numeric_class == 9:
            subclass = "Climbing"
        elif numeric_class == 10:
            subclass = "Club, bar"
        elif numeric_class == 11:
            subclass = "College, university"
        elif numeric_class == 12:
            subclass = "Dance"
        elif numeric_class == 13:
            subclass = "Desert"
        elif numeric_class == 14:
            subclass = "Destruction"
        elif numeric_class == 15:
            subclass = "Drama"
        elif numeric_class == 16:
            subclass = "Drinking"
        elif numeric_class == 17:
            subclass = "Exercise"
        elif numeric_class == 18:
            subclass = "Family"
        elif numeric_class == 19:
            subclass = "Food"
        elif numeric_class == 20:
            subclass = "Glamor, Fashion, Modeling"
        elif numeric_class == 21:
            subclass = "Hiking"
        elif numeric_class == 22:
            subclass = "Horror"
        elif numeric_class == 23:
            subclass = "Hospital"
        elif numeric_class == 24:
            subclass = "Lab experiment"
        elif numeric_class == 25:
            subclass = "Military"
        elif numeric_class == 26:
            subclass = "Misc"
        elif numeric_class == 27:
            subclass = "Monster, Zombies"
        elif numeric_class == 28:
            subclass = "Murder, Dead"
        elif numeric_class == 29:
            subclass = "Music"
        elif numeric_class == 30:
            subclass = "Nudity"
        elif numeric_class == 31:
            subclass = "Outdoor, Nature, Forest"
        elif numeric_class == 32:
            subclass = "Police"
        elif numeric_class == 33:
            subclass = "Prison"
        elif numeric_class == 34:
            subclass = "Robot"
        elif numeric_class == 35:
            subclass = "Romance"
        elif numeric_class == 36:
            subclass = "Sci-fi"
        elif numeric_class == 37:
            subclass = "Sex"
        elif numeric_class == 38:
            subclass = "Smoking"
        elif numeric_class == 39:
            subclass = "Sport, Athletics"
        elif numeric_class == 40:
            subclass = "Super hero"
        elif numeric_class == 41:
            subclass = "Swimming"
        elif numeric_class == 42:
            subclass = "Sword fight"
        elif numeric_class == 43:
            subclass = "Technology"
        elif numeric_class == 44:
            subclass = "Valley, Hills"
        elif numeric_class == 45:
            subclass = "Vehicle"
        elif numeric_class == 46:
            subclass = "Vehicle crash"
        elif numeric_class == 47:
            subclass = "Violence"
        elif numeric_class == 48:
            subclass = "War"
        elif numeric_class == 49:
            subclass = "Weapong"
        elif numeric_class == 50:
            subclass = "Wedding"

        if (numeric_class == 1 or numeric_class == 14 or numeric_class == 47 or numeric_class == 6
                or numeric_class == 42 or numeric_class == 42 or numeric_class == 0 or numeric_class == 7
                or numeric_class == 46):
            superclass = "Action-like movies"
        elif (numeric_class == 2 or numeric_class == 9 or numeric_class == 31 or numeric_class == 3 or
              numeric_class == 13 or numeric_class == 44 or numeric_class == 5 or numeric_class == 21):
            superclass = "Nature-like movies"
        elif numeric_class == 18 or numeric_class == 8:
            superclass = "Family-like movies"
        elif numeric_class == 10 or numeric_class == 50 or numeric_class == 12 or numeric_class == 29:
            superclass = "Party-like movies"
        elif numeric_class == 11 or numeric_class == 23:
            superclass = "Study-like movies"
        elif numeric_class == 16 or numeric_class == 19 or numeric_class == 38:
            superclass = "Bad habits movies"
        elif numeric_class == 35 or numeric_class == 30 or numeric_class == 20 or numeric_class == 37:
            superclass = "Fashion world movies"
        elif numeric_class == 22 or numeric_class == 27 or numeric_class == 28:
            superclass = "Horror-like movies"
        elif (numeric_class == 24 or numeric_class == 36 or numeric_class == 40 or numeric_class == 43 or
              numeric_class == 34):
            superclass = "Sci-fi-like movies"
        elif (numeric_class == 25 or numeric_class == 32 or numeric_class == 33 or numeric_class == 48 or
              numeric_class == 49):
            superclass = "Police-like or war-like movies"
        elif numeric_class == 41 or numeric_class == 17 or numeric_class == 39:
            superclass = "Sport-like movies"
        elif numeric_class == 4:
            superclass = "Animation movie"
        elif numeric_class == 26:
            superclass = "Miscelaneous movie"
        else:
            superclass = "Drama movie"

        return "La película tiene la etiqueta " + subclass + " que está dentro de la súper etiqueta: " + superclass

    @staticmethod
    def write_file(file, message):
        with open(file, 'a') as file:
            file.write(str(message) + '\n')

    @staticmethod
    def format_path(path):
        aux_string = str(path).split('/')
        aux_string = aux_string[len(aux_string) - 1]
        aux_string = aux_string.split('\\')
        image = aux_string[len(aux_string) - 1]
        label = aux_string[len(aux_string) - 2]
        return image, label

    @staticmethod
    def transform_path_for_dataset(path):
        path = str.replace(path, '\\\\', '/')
        path = str.replace(path, '\\', '/')
        if '/' in path:
            path = str.split(path, '/')
        res = path[-1]
        return res